library(sangerseqR)

process_scf_folder <- function(dir_path) {
  # Directory containing the SCF files
  trace_dir <- file.path(dir_path, "HOMO_SAPIENS/FLJ/traces")
  
  # Create a folder to save the CSV files
  output_dir <- "traces"
  dir.create(output_dir, showWarnings = FALSE)
  
  # List all the SCF files in the directory
  scf_files <- list.files(trace_dir, pattern = "\\.scf$")
  
  # Iterate through each SCF file
  for (file_name in scf_files) {
    file_path <- file.path(trace_dir, file_name)
    homoscf <- read.scf(file_path)
    
    # Extract the Sample Points
    sample_points <- as.data.frame(homoscf@sample_points)
    colnames(sample_points) <- c("A", "C", "G", "T")
    
    # Get the Basecall Positions
    basecall_positions <- homoscf@basecall_positions
    
    # Extract the Rows Corresponding to the Basecall Positions
    basecall_sample_points <- sample_points[basecall_positions, ]
    
    # Export to CSV with the same filename but .csv extension
    csv_file_name <- sub("\\.scf$", ".csv", file_name)
    csv_file_path <- file.path(output_dir, csv_file_name)
    write.csv(basecall_sample_points, csv_file_path, row.names = FALSE)
  }
  
  cat(paste("SCF folder", basename(dir_path), "processed and saved as CSV.\n"))
}

# Directory containing the SCF folders
parent_dir <- "extracted_scf"

# List all SCF folders in the parent directory
scf_folders <- list.files(parent_dir, pattern = "scf.homo_sapiens")

# Iterate through each SCF folder and process it
for (scf_folder in scf_folders) {
  scf_folder_path <- file.path(parent_dir, scf_folder)
  process_scf_folder(scf_folder_path)
}

cat("All SCF folders processed and saved as CSV.")