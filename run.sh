#!/bin/bash

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print messages
print_message() {
    echo -e "${2}${1}${NC}"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> logs/reconciliation.log
}

# Function to check if a command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        print_message "Error: $1 is not installed" "$RED"
        exit 1
    fi
}

# Function to check Python version
check_python_version() {
    local python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    local major_version=$(echo $python_version | cut -d. -f1)
    local minor_version=$(echo $python_version | cut -d. -f2)
    
    if [ "$major_version" -lt 3 ] || ([ "$major_version" -eq 3 ] && [ "$minor_version" -lt 8 ]); then
        print_message "Python 3.8 or higher is required. Found version $python_version" "$RED"
        exit 1
    else
        print_message "Python version $python_version detected" "$GREEN"
    fi
}

# Function to check system resources
check_resources() {
    # Check disk space
    local disk_space=$(df -h . | awk 'NR==2 {print $4}' | sed 's/G//')
    if (( $(echo "$disk_space < 1" | bc -l) )); then
        print_message "Warning: Less than 1GB of disk space available" "$YELLOW"
    fi

    # Check memory
    local total_mem=$(sysctl hw.memsize | awk '{print $2}')
    local free_mem=$(vm_stat | awk '/free/ {print $3}' | tr -d '.')
    local available_mem=$((free_mem * 4096))  # Convert pages to bytes
    local mem_percent=$((100 * available_mem / total_mem))
    
    if [ $mem_percent -lt 20 ]; then
        print_message "Warning: Less than 20% of memory available" "$YELLOW"
    fi
}

# Function to create virtual environment
create_venv() {
    if [ ! -d "venv" ]; then
        print_message "Creating virtual environment..." "$BLUE"
        python3 -m venv venv
        if [ $? -ne 0 ]; then
            print_message "Error: Failed to create virtual environment" "$RED"
            exit 1
        fi
        print_message "Virtual environment created successfully" "$GREEN"
    else
        print_message "Virtual environment already exists" "$GREEN"
    fi
}

# Function to activate virtual environment
activate_venv() {
    print_message "Activating virtual environment..." "$BLUE"
    source venv/bin/activate
    if [ $? -ne 0 ]; then
        print_message "Error: Failed to activate virtual environment" "$RED"
        exit 1
    fi
    print_message "Virtual environment activated" "$GREEN"
}

# Function to install dependencies
install_dependencies() {
    print_message "Installing dependencies..." "$BLUE"
    pip install --upgrade pip
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        print_message "Error: Failed to install dependencies" "$RED"
        exit 1
    fi
    print_message "Dependencies installed successfully" "$GREEN"
}

# Function to check for package conflicts
check_package_conflicts() {
    print_message "Checking for package conflicts..." "$BLUE"
    pip check
    if [ $? -ne 0 ]; then
        print_message "Warning: Package conflicts detected" "$YELLOW"
    else
        print_message "No package conflicts found" "$GREEN"
    fi
}

# Function to create necessary directories
create_directories() {
    print_message "Creating necessary directories..." "$BLUE"
    mkdir -p data output logs reconciliation
    if [ $? -ne 0 ]; then
        print_message "Error: Failed to create directories" "$RED"
        exit 1
    fi
    print_message "Directories created successfully" "$GREEN"
}

# Function to backup master files
backup_master_files() {
    if [ -f "data/orders_master.csv" ] || [ -f "data/returns_master.csv" ] || [ -f "data/settlement_master.csv" ]; then
        print_message "Creating backup of master files..." "$BLUE"
        local backup_dir="data/backups/$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$backup_dir"
        cp data/*_master.csv "$backup_dir/" 2>/dev/null
        print_message "Backup created in $backup_dir" "$GREEN"
    fi
}

# Function to cleanup old backups
cleanup_old_backups() {
    print_message "Cleaning up old backups..." "$BLUE"
    find data/backups -type d -mtime +7 -exec rm -rf {} \;
    print_message "Old backups cleaned up" "$GREEN"
}

# Function to check master files
check_master_files() {
    print_message "Checking master files..." "$BLUE"
    local files=("data/orders_master.csv" "data/returns_master.csv" "data/settlement_master.csv")
    local missing_files=()
    
    for file in "${files[@]}"; do
        if [ ! -f "$file" ]; then
            missing_files+=("$file")
        else
            local size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file")
            print_message "Found $file (Size: $(numfmt --to=iec $size))" "$GREEN"
        fi
    done
    
    if [ ${#missing_files[@]} -eq 0 ]; then
        print_message "All master files present" "$GREEN"
    else
        print_message "Starting with fresh data (missing files: ${missing_files[*]})" "$YELLOW"
    fi
}

# Main execution
print_message "Starting Order Reconciliation Dashboard Setup..." "$BLUE"

# Check prerequisites
check_command python3
check_command pip
check_python_version
check_resources

# Create and activate virtual environment
create_venv
activate_venv

# Install dependencies
install_dependencies
check_package_conflicts

# Create directories
create_directories

# Backup and check master files
backup_master_files
cleanup_old_backups
check_master_files

# Run the application
print_message "Starting the application..." "$BLUE"
streamlit run src/app.py

# Deactivate virtual environment when done
deactivate

print_message "Thank you for using the Order Reconciliation Dashboard!" "$GREEN" 