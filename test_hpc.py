import socket
import os
import platform
import subprocess

def test_hpc_connection():
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Hostname: {socket.gethostname()}")
    
    # Try to ping iitj.ac.in and hpc.iitj.ac.in
    hosts_to_test = ["iitj.ac.in", "hpc.iitj.ac.in", "login.hpc.iitj.ac.in"]
    
    for host in hosts_to_test:
        print(f"\n--- Testing connection to {host} ---")
        param = '-n' if platform.system().lower()=='windows' else '-c'
        command = ['ping', param, '1', host]
        
        try:
            output = subprocess.run(command, capture_output=True, text=True, timeout=5)
            if output.returncode == 0:
                print(f"Success! Can reach {host}")
                print(output.stdout.split('\n')[2]) # Print a line from the ping output
            else:
                print(f"Failed to reach {host}")
        except Exception as e:
            print(f"Error testing {host}: {e}")

if __name__ == "__main__":
    test_hpc_connection()
