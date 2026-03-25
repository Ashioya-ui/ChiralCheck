import subprocess
import sys

def run(command):
    """Executes a shell command and prints the output."""
    print(f"\n> {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr and result.returncode != 0:
        print(f"Warning/Error: {result.stderr.strip()}")
    return result.returncode

def main():
    print("🚀 Initiating Automatic Git Fix & Force Push...")
    
    # 1. Abort the currently stuck rebase process
    print("\n--- Step 1: Cleaning up stuck git state ---")
    run("git rebase --abort")
    
    # 2. Ensure all your current local files are staged
    print("\n--- Step 2: Staging local files ---")
    run("git add .")
    
    # 3. Commit any lingering changes
    print("\n--- Step 3: Committing ---")
    run('git commit -m "feat: add Epistemic Immune System and force sync local state"')
    
    # 4. Force push to GitHub (Overwrite remote with local)
    print("\n--- Step 4: Force Pushing to GitHub ---")
    print("⚠️  Note: This overwrites the GitHub repo with your exact local files.")
    success = run("git push -f origin main")
    
    if success == 0:
        print("\n✅ Success! Your local code has been forcefully pushed to GitHub.")
    else:
        print("\n❌ Push failed. Check your internet connection or repository permissions.")

if __name__ == "__main__":
    main()