from tsr.utils import shell_exec

stdout, stderr = shell_exec("python3 -m black .")
print(stdout)
print(stderr)

print("Checking Flake 8")
stdout, stderr = shell_exec("python3 -m flake8 .")
print(stdout)
print(stderr)
