from __future__ import annotations

import argparse
import platform
import shutil
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import pandas as pd
from expidite_rpi.core import api, file_naming

from choice_assay.rpi.choice_assay_pose_processor import (
    DEFAULT_CHOICE_ASSAY_POSE_PROCESSOR_CFG,
    ChoiceAssayPoseProcessor,
    ChoiceAssayPoseProcessorCfg,
)

CONTAINER_NAME = "expidite-choiceassay-trapcam"
TYPE_ID = "CAVIDEO"
PREFIX = f"V3_{TYPE_ID}_"
SUFFIX = ".mp4"
# OUTPUT_PREFIX_LEN = len("V3_CAVIDEO_d83add1a11c5_00_00_20260317")
DEFAULT_LOCAL_CACHE_DIRNAME = "choice_assay_video_cache"
DEFAULT_PREFETCH_AHEAD = 6
DEFAULT_PREFETCH_WAIT_LOG_INTERVAL_SECONDS = 30.0
DEFAULT_PREFETCH_HARD_TIMEOUT_SECONDS: float | None = None
DEFAULT_MAX_WORKERS = 2


def detect_default_roots() -> tuple[Path, Path]:
    """Return default NAS and temp roots based on host platform."""
    running_on_linux = "Linux" in platform.platform()
    if running_on_linux:
        return Path("~/bee-ops-disk/").expanduser(), Path("/tmp/")

    nas_root = Path("B://")
    tmp_root = Path.home() / "AppData" / "Local" / "Temp"
    if not nas_root.exists():
        print("Warning: B: drive not found, falling back to local alternative")
        nas_root = Path.home() / "bee-ops-disk"
    return nas_root, tmp_root


def list_processed_videos(processed_log_path: Path) -> set[str]:
    """List videos already handled in prior runs, including no-detection outcomes."""
    if not processed_log_path.exists():
        return set()

    df = pd.read_csv(processed_log_path)
    required_cols = {"video_filename", "status"}
    if not required_cols.issubset(df.columns):
        print(f"Warning: {processed_log_path} missing required columns {required_cols}; ignoring log.")
        return set()

    terminal_statuses = {
        "processed_with_detection",
        "processed_no_detection",
        "missing_prefetch",
        "missing_processing",
    }
    done = df.loc[df["status"].isin(terminal_statuses), "video_filename"].dropna().astype(str)
    return set(done.tolist())


def record_video_attempt(
    processed_log_path: Path,
    video_fname: str,
    status: str,
    rows_saved: int = 0,
    note: str = "",
) -> None:
    """Append a processing outcome record so reruns can skip already handled videos."""
    record = pd.DataFrame(
        [
            {
                "video_filename": video_fname,
                "status": status,
                "rows_saved": rows_saved,
                "processed_at_utc": datetime.now(UTC).isoformat(),
                "note": note,
            }
        ]
    )
    record.to_csv(
        processed_log_path,
        mode="a",
        header=not processed_log_path.exists(),
        index=False,
    )


def save_results_to_csv(
    results: pd.DataFrame,
    output_dir: Path,
    video_fname: str,
) -> int:
    """Save the results to a CSV file and return the number of rows written.

    We need to do 2 specific bits of mapping:
    - we need to match the output from exipidte, so we need to add the appropriate index columns
    - we need to match the filename, mapping from a prefix of "V3_CAVIDEO_d83add1a11d5_00_00_20260624" to
      "V3_CAPOSE_d83add1a11d5_20260624".
    """
    if results.empty:
        print("No results to save.")
        return 0

    parts = file_naming.parse_record_filename(video_fname)
    timestamp: datetime = parts["timestamp"]
    journal_fname = f"V3_CAPOSE_{parts['device_id']}_{timestamp.strftime('%Y%m%d')}.csv"

    output_csv = output_dir / f"{journal_fname}.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Add the index columns to match the expected output format
    # We add columns for all of api.ALL_RECORD_ID_FIELDS
    for field in api.ALL_RECORD_ID_FIELDS:
        if field not in results.columns:
            if field == api.RECORD_ID.VERSION.value:
                results[field] = "V3"
            elif field == api.RECORD_ID.DATA_TYPE_ID.value:
                results[field] = "CAPOSE"
            elif field in {api.RECORD_ID.SENSOR_INDEX.value, api.RECORD_ID.STREAM_INDEX.value}:
                results[field] = 0
            elif field == api.RECORD_ID.DEVICE_ID.value:
                results[field] = parts["device_id"]
            elif field == api.RECORD_ID.TIMESTAMP.value:
                results[field] = api.utc_to_iso_str(timestamp)
            else:
                results[field] = None

    if output_csv.exists():
        results.to_csv(output_csv, mode="a", header=False, index=False)
    else:
        results.to_csv(output_csv, index=False)

    return len(results)


def create_cfg() -> ChoiceAssayPoseProcessorCfg:
    """Create a configuration object for the choice assay pose processor."""
    cfg = DEFAULT_CHOICE_ASSAY_POSE_PROCESSOR_CFG
    marked_up_output = cfg.outputs[1]
    marked_up_output.sample_probability = 0
    return replace(cfg, outputs=[cfg.outputs[0], marked_up_output])


def stage_video_to_local(video_path: Path, local_cache_dir: Path) -> tuple[Path, float, int, int]:
    """Copy video from NAS to local cache and return path, copy_seconds, remote_size, local_size."""
    if not video_path.exists():
        msg = f"Remote video not found: {video_path}"
        raise FileNotFoundError(msg)

    local_path = local_cache_dir / video_path.name
    remote_size = video_path.stat().st_size
    t_copy_start = perf_counter()
    if not local_path.exists():
        shutil.copy2(video_path, local_path)
    copy_seconds = perf_counter() - t_copy_start
    local_size = local_path.stat().st_size
    return local_path, copy_seconds, remote_size, local_size


def run_rerun_processing(
    files_to_process: Path,
    video_src_dir: Path,
    output_dir: Path,
    file_filter: str | None = None,
) -> dict[str, int]:
    """Run processing over all listed videos with JIT prefetch staging.

    Args:
        files_to_process: Path to a CSV or text file containing one video filename per line.
        video_src_dir: Directory containing the source video files.
        output_dir: Directory to save processed results.
        file_filter: Optional substring to filter video filenames.  Do not use wild cards.
        This is a simple substring match.

    """
    _, tmp_root = detect_default_roots()
    local_cache_dir = tmp_root / DEFAULT_LOCAL_CACHE_DIRNAME
    prefetch_ahead = DEFAULT_PREFETCH_AHEAD
    prefetch_wait_log_interval_seconds = DEFAULT_PREFETCH_WAIT_LOG_INTERVAL_SECONDS
    prefetch_hard_timeout_seconds = DEFAULT_PREFETCH_HARD_TIMEOUT_SECONDS
    max_workers = DEFAULT_MAX_WORKERS

    output_dir.mkdir(parents=True, exist_ok=True)
    local_cache_dir.mkdir(parents=True, exist_ok=True)
    processed_log_path = output_dir / "processed_videos_log.csv"

    processed_videos = set(list_processed_videos(processed_log_path))
    raw_file_list = set(pd.read_csv(files_to_process, header=None)[0].to_list())
    videos_to_process = sorted(raw_file_list - processed_videos)
    if file_filter is not None:
        videos_to_process = [fname for fname in videos_to_process if file_filter in fname]
    videos_to_process = [video_src_dir / fname for fname in videos_to_process]

    print(f"Found {len(videos_to_process)} new videos to process. {len(processed_videos)} already handled.")
    print(f"JIT local cache directory: {local_cache_dir.resolve()}")
    print(f"Prefetch ahead: {prefetch_ahead} files")
    print(f"Processing log: {processed_log_path.resolve()}")

    processor = ChoiceAssayPoseProcessor(create_cfg(), 0)
    start_time = datetime.now(UTC)

    def submit_prefetch(
        executor: ThreadPoolExecutor,
        queue: dict[int, Future],
        submitted_at: dict[int, float],
        index: int,
    ) -> None:
        if 0 <= index < len(videos_to_process) and index not in queue:
            remote_path = Path(videos_to_process[index])
            queue[index] = executor.submit(stage_video_to_local, remote_path, local_cache_dir)
            submitted_at[index] = perf_counter()

    def prefetch_queue_snapshot(
        queue: dict[int, Future],
        submitted_at: dict[int, float],
        limit: int = 6,
    ) -> str:
        pending = []
        now = perf_counter()
        for idx in sorted(queue.keys())[:limit]:
            fut = queue[idx]
            age = now - submitted_at.get(idx, now)
            if fut.done():
                state = "done"
            elif fut.running():
                state = "running"
            else:
                state = "pending"
            pending.append(f"{idx}:{state}:{age:.1f}s")
        suffix = " ..." if len(queue) > limit else ""
        return ", ".join(pending) + suffix

    prefetch_futures: dict[int, Future] = {}
    prefetch_submitted_at: dict[int, float] = {}
    copy_seconds_total = 0.0
    prefetch_wait_seconds_total = 0.0
    inference_seconds_total = 0.0
    skipped_missing_prefetch = 0
    skipped_missing_processing = 0
    processed_success = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for idx in range(min(prefetch_ahead, len(videos_to_process))):
            submit_prefetch(pool, prefetch_futures, prefetch_submitted_at, idx)

        for i in range(len(videos_to_process)):
            submit_prefetch(pool, prefetch_futures, prefetch_submitted_at, i + prefetch_ahead)

            remote_video = Path(videos_to_process[i])
            future = prefetch_futures[i]
            wait_start = perf_counter()
            local_video: Path | None = None

            while True:
                try:
                    local_video, copy_seconds, remote_size, _local_size = future.result(
                        timeout=prefetch_wait_log_interval_seconds
                    )
                    prefetch_wait_seconds_total += perf_counter() - wait_start
                    break
                except TimeoutError as err:
                    wait_so_far = perf_counter() - wait_start
                    snapshot = prefetch_queue_snapshot(prefetch_futures, prefetch_submitted_at)
                    print(
                        f"Waiting on prefetch index={i} file={remote_video.name} for {wait_so_far:.1f}s"
                        f" | queue={len(prefetch_futures)} [{snapshot}]"
                    )
                    if (
                        prefetch_hard_timeout_seconds is not None
                        and wait_so_far > prefetch_hard_timeout_seconds
                    ):
                        msg = (
                            f"Prefetch wait exceeded {prefetch_hard_timeout_seconds}s for "
                            f"{remote_video.name} | queue={len(prefetch_futures)} [{snapshot}]"
                        )
                        raise TimeoutError(msg) from err
                except FileNotFoundError as err:
                    prefetch_wait_seconds_total += perf_counter() - wait_start
                    skipped_missing_prefetch += 1
                    record_video_attempt(
                        processed_log_path,
                        remote_video.name,
                        status="missing_prefetch",
                        rows_saved=0,
                        note=str(err),
                    )
                    print(
                        f"Skipping missing file during prefetch ({skipped_missing_prefetch}): "
                        f"{remote_video.name} | reason={err}"
                    )
                    break

            prefetch_futures.pop(i, None)
            prefetch_submitted_at.pop(i, None)

            if local_video is None:
                continue

            copy_seconds_total += copy_seconds

            if not local_video.exists():
                skipped_missing_processing += 1
                msg = f"Local cache file missing before processing: {local_video}"
                record_video_attempt(
                    processed_log_path,
                    remote_video.name,
                    status="missing_processing",
                    rows_saved=0,
                    note=msg,
                )
                print(
                    "Skipping missing local cache file during processing "
                    f"({skipped_missing_processing}): {local_video}"
                )
                continue

            t_infer_start = perf_counter()
            try:
                df = processor.process_video_file(local_video)
                inference_seconds = perf_counter() - t_infer_start
                inference_seconds_total += inference_seconds
                rows_saved = save_results_to_csv(df, output_dir, remote_video.name)
                status = "processed_with_detection" if rows_saved > 0 else "processed_no_detection"
                record_video_attempt(
                    processed_log_path,
                    remote_video.name,
                    status=status,
                    rows_saved=rows_saved,
                )
                processed_success += 1
            except FileNotFoundError as err:
                skipped_missing_processing += 1
                record_video_attempt(
                    processed_log_path,
                    remote_video.name,
                    status="missing_processing",
                    rows_saved=0,
                    note=str(err),
                )
                print(
                    f"Skipping file missing during processing ({skipped_missing_processing}): "
                    f"{remote_video.name} | reason={err}"
                )
                continue
            finally:
                try:
                    local_video.unlink(missing_ok=True)
                except OSError as err:
                    print(f"Warning: failed to remove cache file {local_video}: {err}")

            elapsed = (datetime.now(UTC) - start_time).total_seconds()
            n = i + 1
            print(
                f"{n}/{len(videos_to_process)} @ {elapsed / n:.1f} secs/video"
                f" | infer={inference_seconds:.2f}s"
                f" | copy={copy_seconds:.2f}s ({remote_size / 1e6:.1f}MB)"
                f" | avg_copy={copy_seconds_total / n:.2f}s"
                f" | avg_wait={prefetch_wait_seconds_total / n:.2f}s"
                f" | avg_infer={inference_seconds_total / n:.2f}s"
                f" | ok={processed_success}"
                f" | miss_prefetch={skipped_missing_prefetch}"
                f" | miss_process={skipped_missing_processing}"
                f" | cache_queue={len(prefetch_futures)}"
                f" | processed {remote_video.name} | output -> {output_dir.resolve()}"
            )

    print(
        f"Completed rerun: ok={processed_success}, "
        f"missing_in_prefetch={skipped_missing_prefetch}, "
        f"missing_in_processing={skipped_missing_processing}"
    )

    return {
        "processed_success": processed_success,
        "missing_in_prefetch": skipped_missing_prefetch,
        "missing_in_processing": skipped_missing_processing,
        "already_handled": len(processed_videos),
        "queued": len(videos_to_process),
    }


def parse_args() -> argparse.Namespace:
    nas_root, _ = detect_default_roots()
    default_video_src_dir = nas_root / "azure" / "choice_assay" / "expidite-choiceassay-trapcam"
    default_output_dir = nas_root / "results" / "choice_assay_rerun"
    default_files_list = Path(__file__).with_name("files_to_process.csv")

    parser = argparse.ArgumentParser(description="Re-run choice assay video processing in standalone mode.")
    parser.add_argument(
        "--files-to-process",
        type=Path,
        default=default_files_list,
        help="Path to CSV/txt containing one video filename per line.",
    )
    parser.add_argument("--video-src-dir", type=Path, default=default_video_src_dir)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument(
        "--file-filter",
        type=str,
        default=None,
        help="Optional substring to filter video filenames.  Do not use wild cards.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Processing {PREFIX} files from '{CONTAINER_NAME}' to {args.video_src_dir.resolve()}")

    run_rerun_processing(
        files_to_process=args.files_to_process,
        video_src_dir=args.video_src_dir,
        output_dir=args.output_dir,
        file_filter=args.file_filter
    )


if __name__ == "__main__":
    main()
