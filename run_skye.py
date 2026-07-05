"""
SKYE Platform Launcher
======================
Central entry point to verify, benchmark, and launch the SKYE Edge AI system.
"""

import sys
import subprocess
import time
import webbrowser

def main():
    print("=" * 70)
    print("        WELCOME TO THE SKYE ON-AIRCRAFT EDGE AI PLATFORM")
    print("=" * 70)
    print("\nStarting the Real-Time Flight Diagnostics Dashboard...")
    print("Checking dependencies and trained models...")
    cmd = [sys.executable, "-m", "dashboard.app"]
    
    try:
        proc = subprocess.Popen(cmd)
        time.sleep(2.0)
        
        url = "http://127.0.0.1:8050"
        print(f"\nDashboard launched successfully!")
        print(f"Access URL: {url}")
        print("Press Ctrl+C in this terminal to terminate the simulation.\n")
        webbrowser.open(url)
        proc.wait()
        
    except KeyboardInterrupt:
        print("\nShutting down SKYE Platform...")
        proc.terminate()
        proc.wait()
        print("Goodbye!")
    except Exception as e:
        print(f"Error starting dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
