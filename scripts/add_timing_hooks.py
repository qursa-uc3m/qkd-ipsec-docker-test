#!/usr/bin/env python3

import os
import re
import sys

def main():
    strongswan_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    timing_log_file = "/tmp/plugin_timing.csv"
    
    print(f"StrongSwan Timing Hooks Installer")
    print(f"==================================")
    print(f"Working in directory: {strongswan_dir}")
    
    # Plugin configurations
    plugins = [
        {
            "file": "src/libstrongswan/plugins/openssl/openssl_ec_diffie_hellman.c",
            "struct": "private_openssl_ec_diffie_hellman_t",
            "timing_member": "group",
            "plugin_name": "openssl"
        },
        {
            "file": "src/libstrongswan/plugins/openssl/openssl_x_diffie_hellman.c",
            "struct": "private_key_exchange_t",
            "timing_member": "ke",
            "plugin_name": "openssl"
        },
        {
            "file": "src/libstrongswan/plugins/oqs/oqs_kem.c",
            "struct": "private_oqs_kem_t",
            "timing_member": "method",
            "plugin_name": "oqs"
        }
    ]
    
    # Process each plugin
    for plugin in plugins:
        file_path = os.path.join(strongswan_dir, plugin["file"])
        if not os.path.isfile(file_path):
            print(f"WARNING: File not found: {file_path}")
            continue
        
        install_timing_hooks(file_path, plugin, timing_log_file)
    
    # Initialize the timing log file
    create_log_file(timing_log_file)
    
    print("Timing hooks installation completed")

def install_timing_hooks(file_path, plugin, timing_log_file):
    """Install timing hooks in a strongSwan plugin file"""
    print(f"Processing {os.path.basename(file_path)}...")
    
    # Read the file content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Perform all modifications
    modified_content = content
    
    # 1. Add required includes (only once at the top)
    if not all(inc in modified_content for inc in ["#include <time.h>", "#include <stdio.h>", "#include <sys/time.h>"]):
        modified_content = add_includes(modified_content)
    
    # 2. Add struct member
    if "struct timeval create_time;" not in modified_content:
        modified_content = add_struct_member(modified_content, plugin["struct"])
    
    # 3. Add log_time function
    if "static void log_time" not in modified_content:
        modified_content = add_log_time_function(modified_content, plugin, timing_log_file)
    
    # 4. Add log_time call in destroy method
    if "log_time(this);" not in modified_content:
        modified_content = add_log_time_call(modified_content)
    
    # 5. Add initialization in create function
    modified_content = add_initialization(modified_content, os.path.basename(file_path))
    
    # Write the modified content back to the file if changes were made
    if modified_content != content:
        with open(file_path, 'w') as f:
            f.write(modified_content)

def add_includes(content):
    """Add necessary includes at the beginning of the file"""
    first_include = re.search(r'#include\s+<[^>]+>', content)
    if first_include:
        new_includes = "\n#include <time.h>   /* For timing measurements */\n#include <stdio.h>  /* For file operations */\n#include <sys/time.h> /* For microsecond precision */"
        return content[:first_include.end()] + new_includes + content[first_include.end():]
    return content

def add_struct_member(content, struct_name):
    """Add struct timeval member to the specified struct"""
    struct_pattern = f"struct {struct_name} {{.*?}};"
    match = re.search(struct_pattern, content, re.DOTALL)
    if match:
        struct_def = match.group(0)
        updated_struct = struct_def.replace("};", "    struct timeval create_time;\n};")
        return content.replace(struct_def, updated_struct)
    return content

def add_log_time_function(content, plugin, timing_log_file):
    """Add the log_time function after struct definition"""
    struct_pattern = f"struct {plugin['struct']} {{.*?}};"
    match = re.search(struct_pattern, content, re.DOTALL)
    if match:
        log_func = f"""
#define QKD_TIMING_LOG "{timing_log_file}"

static void log_time({plugin['struct']} *this)
{{
    struct timeval destroy_time;
    FILE *fp;
    
    gettimeofday(&destroy_time, NULL);
    
    /* Open file in append mode */
    fp = fopen(QKD_TIMING_LOG, "a");
    if (fp == NULL)
    {{
        DBG1(DBG_LIB, "{plugin['plugin_name']}_plugin: Could not open timing log file: %s", QKD_TIMING_LOG);
        return;
    }}
    
    /* Write timing data */
    fprintf(fp, "%d,%ld,%ld,%ld,%ld\\n", 
            this->{plugin['timing_member']}, 
            (long)this->create_time.tv_sec,
            (long)this->create_time.tv_usec,
            (long)destroy_time.tv_sec,
            (long)destroy_time.tv_usec
            );
    
    fclose(fp);
    
    DBG1(DBG_LIB, "{plugin['plugin_name']}_plugin: Logged timing event for method %d", this->{plugin['timing_member']});
}}
"""
        return content.replace(match.group(0), match.group(0) + log_func)
    return content

def add_log_time_call(content):
    """Add log_time call to the destroy method"""
    destroy_pattern = r'(METHOD\(.*?,\s*destroy,.*?)free\(this\);'
    match = re.search(destroy_pattern, content, re.DOTALL)
    if match:
        modified = match.group(1) + "log_time(this);\n\tfree(this);"
        return content.replace(match.group(0), modified)
    return content

def add_initialization(content, filename):
    """Add initialization code to the create function"""
    # First, remove any misplaced initialization
    content = re.sub(r'gettimeofday\(&this->create_time, NULL\);(\s+return NULL;)', r'\1', content)
    
    # Different handling based on file type
    if "oqs_kem.c" in filename:
        # For OQS, add after memset
        if "memset(this->shared_secret" in content:
            pattern = r'(memset\(this->shared_secret.*?;)'
            replacement = r'\1\n\t/* Record creation time */\n\tgettimeofday(&this->create_time, NULL);'
            content = re.sub(pattern, replacement, content)
    else:
        # For other plugins, add before main return statement
        pattern = r'(return &this->public;)'
        replacement = r'\tgettimeofday(&this->create_time, NULL);\n\t\1'
        content = re.sub(pattern, replacement, content)
    
    # Fix any indentation issues
    content = re.sub(r'(^|\s)tgettimeofday', r'\1\tgettimeofday', content, flags=re.MULTILINE)
    content = re.sub(r'\\tgettimeofday', r'\tgettimeofday', content)
    
    return content

def create_log_file(file_path):
    """Create the timing log file if it doesn't exist"""
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write("method,create_time_sec,create_time_usec,destroy_time_sec,destroy_time_usec\n")
        os.chmod(file_path, 0o666)
        print(f"Created timing log file: {file_path}")
    else:
        print(f"Timing log file already exists: {file_path}")

if __name__ == "__main__":
    main()