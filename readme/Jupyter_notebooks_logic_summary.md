# Jupyter Notebooks Logic Summary - High-Level Pseudo-Code

## Overview
This document summarizes the logic and workflow of three interconnected Jupyter notebooks used for bee behavior analysis using DeepLabCut (DLC) computer vision and subsequent data processing.

## 1. DLC Data Extraction Notebook (1_DLC_data_extraction.ipynb)

### Purpose
Processes raw video files using DeepLabCut to extract body part positions and generate CSV files with pose estimation data.

### Main Workflow

#### 1.1 Environment Setup
```
INSTALL dependencies:
    - CUDA 11.8 for GPU acceleration
    - DeepLabCut with TensorFlow support

MOUNT Google Drive for cloud-based processing

SET project variables:
    - ProjectFolderName = 'Beecam_v4-Anna-2024-06-25'
    - VideoType = 'mp4'
    - path_config_file = DLC project configuration path
```

#### 1.2 Video Discovery and Processing Management
```
DEFINE experiment parameters:
    - ROUND = experiment round number (e.g., 22)
    - PROJECT_DATA_FOLDER = base directory for experimental data
    - folder_to_analyse = specific round folder to process

CREATE video inventory system:
    FUNCTION list_all_videos(source_dir):
        Walk through directory structure
        Find all .mp4 files
        Extract relative paths from round folder
        RETURN list of video paths
    
    FUNCTION list_processed_videos(source_dir):
        Search for existing DLC output files (containing "DLC_resnet50")
        Extract corresponding video names
        RETURN list of already processed videos
    
    FUNCTION get_unprocessed_videos(all_videos, processed_videos):
        Compare total videos vs processed videos
        RETURN list of videos still needing analysis
```

#### 1.3 Batch Video Processing
```
FOR each RPi folder in the round:
    INCREMENT RPi counter
    GET list of video files in Videos subdirectory
    
    FOR each video file:
        CHECK if video already processed (skip if yes)
        
        IF not processed:
            CALL deeplabcut.analyze_videos():
                - Input: video file path, config file
                - Output: CSV file with pose data
                - Parameters: save_as_csv=True
            
            LOG progress (video X of Y processed)
```

#### 1.4 Output Structure
```
GENERATES CSV files containing:
    - Frame numbers
    - Body part coordinates (x, y)
    - Confidence scores for each detection
    - Separate file for each processed video
```

---

## 2. Data Extraction from CSVs Notebook (2_Extraction_of_data_from_csvs.ipynb)

### Purpose  
Processes DLC-generated CSV files to extract behavioral data, focusing on proboscis activity detection and classification.

### Main Workflow

#### 2.1 Setup and File Management
```
MOUNT Google Drive

DEFINE experiment parameters:
    - FOLDER_NAME = experiment directory name
    - output_directory = path for processed datasets
    - PATH_EXPORT = final output CSV path

CREATE OR LOAD master dataset:
    IF dataset exists:
        READ existing proboscis data CSV
    ELSE:
        CREATE new CSV with predefined columns:
            [Frame, body_part_coordinates, Time_data, Tube, RPi, Video, Round, Behaviour]
```

#### 2.2 CSV Discovery and Processing Status
```
FUNCTION list_all_csvs(source_dir):
    Walk through experiment directory
    Find all DLC output CSV files  
    RETURN complete list of CSV files

IDENTIFY processed vs unprocessed CSVs:
    Compare existing dataset entries with available CSV files
    Determine which videos need behavioral analysis
```

#### 2.3 Behavioral Data Extraction Pipeline
```
FOR each unprocessed CSV file:
    EXTRACT metadata from filename:
        - Tube side (left/right)
        - Recording timestamp
        - RPi identifier
        - Round number
    
    LOAD and CLEAN CSV data:
        Remove DLC header rows (rows 0,1)
        Rename columns to standard format
        Convert data types appropriately
    
    CALCULATE temporal information:
        Time_seconds = Frame * 0.2 (assuming 5fps)
        Time_in_experiment = relative time from session start
        Clock_time = absolute timestamp
    
    FILTER for proboscis activity:
        SELECT frames WHERE:
            (Tube_prob_likelihood > 0.6 OR End_prob_likelihood > 0.6)
            AND End_prob_y >= 10
            AND Tube_prob_y >= 10
    
    CLASSIFY behavior:
        FUNCTION get_behaviour(row):
            IF (Tube_prob_likelihood >= 0.8 AND End_prob_likelihood >= 0.8):
                RETURN "Drinking"
            ELIF (Tube_prob_likelihood >= 0.8 AND End_prob_likelihood < 0.8):
                RETURN "Drinking_no_end"  
            ELIF (Tube_prob_likelihood < 0.8 AND End_prob_likelihood >= 0.8):
                RETURN "Prob_out"
            ELSE:
                RETURN "No_prob"
    
    APPEND processed data to master CSV
```

#### 2.4 Output Generation
```
GENERATE multiple output datasets:
    - all_data.csv: Complete DLC data with temporal information
    - proboscis_data.csv: Filtered frames showing proboscis activity  
    - drinking_data.csv: Subset containing only drinking behaviors
```

---

## 3. Data Visualization Notebook (3_Data_visualisation.ipynb)

### Purpose
Converts frame-level proboscis data into bout-based analysis and creates visualizations for behavioral patterns.

### Main Workflow

#### 3.1 Environment Setup and Data Loading
```
MOUNT Google Drive

LOAD R environment and required packages:
    - ggplot2 (visualization)
    - dplyr (data manipulation)  
    - lubridate (time handling)
    - tidyverse (data science toolkit)

LOAD processed proboscis data:
    READ proboscis_data.csv from previous notebook
    VERIFY data completeness and round numbers
```

#### 3.2 Data Preprocessing and Time Conversion
```
STANDARDIZE time variables:
    EXTRACT time portion from Time_in_experiment field
    CONVERT to R-compatible time format using lubridate::hms()
    CREATE numeric time representation (Time_min)
    
CLEAN and ORGANIZE data:
    REMOVE any NA values
    CONVERT RPi identifiers to factors
    STANDARDIZE RPi naming convention:
        IF RPi name ends with '_': use 4th character
        ELSE: use 4th and 5th characters
```

#### 3.3 Bout Detection and Analysis
```
BOUT IDENTIFICATION algorithm:
    GROUP consecutive frames by:
        - Same RPi
        - Same tube
        - Same behavior type
        - Temporal proximity (within threshold)
    
    FOR each behavioral bout:
        CALCULATE bout duration
        DETERMINE bout start and end times
        CLASSIFY bout type (drinking, proboscis_out, etc.)
        EXTRACT bout intensity metrics
```

#### 3.4 Statistical Analysis and Aggregation
```
GENERATE summary statistics:
    - Total feeding time per RPi/tube/round
    - Number of feeding bouts
    - Average bout duration  
    - Feeding frequency patterns
    - Temporal distribution of activity

CREATE experimental comparisons:
    - Between different treatments/rounds
    - Left vs right tube preferences  
    - Individual bee behavioral patterns
    - Time-course analysis within experiments
```

#### 3.5 Visualization Generation
```
CREATE multiple plot types:
    
    TIMELINE PLOTS:
        X-axis: Time in experiment
        Y-axis: RPi/individual identifier
        Color: Behavior type
        Show continuous feeding activity over time
    
    SUMMARY BAR PLOTS:
        Compare total activity between conditions
        Show feeding preferences (left vs right)
        Display bout frequency distributions
    
    HEATMAPS:
        Temporal activity patterns
        Individual behavioral profiles
        Treatment effect comparisons
    
    STATISTICAL PLOTS:
        Box plots for duration comparisons
        Scatter plots for correlation analysis
        Time series for temporal trends
```

#### 3.6 Export and Reporting
```
SAVE processed bout data:
    - bout_data.csv: Aggregated behavioral bout information
    - summary_statistics.csv: Experimental summary metrics
    
EXPORT visualizations:
    - High-resolution plots for publication
    - Interactive plots for exploration
    - Summary report with key findings
```

---

## Integration and Workflow Summary

### Complete Pipeline
```
1. VIDEO CAPTURE → Beecam system records bee behavior videos

2. DLC PROCESSING → Notebook 1 extracts body part positions
   Input: Raw MP4 videos
   Output: CSV files with pose coordinates

3. BEHAVIOR EXTRACTION → Notebook 2 identifies feeding behaviors  
   Input: DLC CSV files
   Output: Labeled behavioral data

4. ANALYSIS & VISUALIZATION → Notebook 3 creates bout analysis and plots
   Input: Behavioral data  
   Output: Statistical summaries and visualizations
```

### Key Features

#### Data Management
- **Incremental Processing**: Only analyzes new/unprocessed videos
- **Batch Processing**: Handles entire experimental rounds automatically  
- **Error Handling**: Skips corrupted or empty files
- **Progress Tracking**: Reports processing status throughout

#### Behavioral Classification
- **Proboscis Detection**: Uses confidence thresholds for reliable detection
- **Behavior Categories**: Distinguishes drinking, proboscis extension, and searching
- **Temporal Context**: Considers bout duration and patterns

#### Analysis Capabilities
- **Multi-level Analysis**: Frame-level, bout-level, and experiment-level metrics
- **Comparative Studies**: Between treatments, individuals, and time periods
- **Statistical Rigor**: Appropriate handling of temporal dependencies

#### Flexibility
- **Configurable Parameters**: Adjustable thresholds and analysis parameters
- **Multiple Output Formats**: CSV data and various visualization types
- **Scalable Processing**: Handles large datasets efficiently