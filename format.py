import subprocess

print(subprocess.check_output("python3 -m black .".split()))
print("Checking Flake 8")
print(subprocess.check_output("python3 -m flake8 .".split()))
