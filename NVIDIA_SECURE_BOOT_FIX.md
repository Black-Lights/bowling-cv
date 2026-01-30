# NVIDIA Driver Secure Boot Fix Guide

## Problem Diagnosis

Your system has:
- ✓ NVIDIA RTX 3060 Mobile GPU (detected)
- ✓ NVIDIA Driver 550.163.01 (installed)
- ✓ CUDA Toolkit 12.0 (installed)
- ✓ NVIDIA kernel modules built for kernel 6.14.0-33 (via DKMS)
- ✗ **Secure Boot ENABLED** - blocking unsigned kernel modules

**Error**: `Key was rejected by service` when loading nvidia module

## Root Cause

Secure Boot is preventing the NVIDIA kernel module from loading because it's not signed with a trusted key. This is a security feature in UEFI systems.

## Solution Options

### Option 1: Disable Secure Boot (RECOMMENDED - Easiest)

This is the simplest and most common solution for NVIDIA drivers.

**Steps:**
1. Reboot your computer
2. Enter BIOS/UEFI setup (press **F2**, **F10**, **DEL**, or **ESC** during boot - varies by manufacturer)
   - For HP Victus: Usually **F10** or **ESC**
3. Navigate to the **Security** tab
4. Find **Secure Boot** option
5. Change it to **Disabled**
6. **Save and Exit** (usually F10)
7. Boot back into Linux
8. Verify the fix:
   ```bash
   sudo modprobe nvidia
   nvidia-smi
   ```

**After Disabling Secure Boot:**
```bash
# Load the NVIDIA module
sudo modprobe nvidia
sudo modprobe nvidia-uvm

# Verify it works
nvidia-smi

# Should show your RTX 3060 Mobile GPU with driver info
```

### Option 2: Sign the NVIDIA Module (Advanced)

This keeps Secure Boot enabled but requires signing the kernel module.

**Steps:**
```bash
# Run the signing script
sudo ./sign_nvidia_module.sh

# This will:
# 1. Generate a Machine Owner Key (MOK)
# 2. Ask you to set a password
# 3. Require a reboot
# 4. During boot, you'll enroll the key in MOK Manager
# 5. Sign all NVIDIA modules
```

**Detailed Process:**
1. Run: `sudo ./sign_nvidia_module.sh`
2. Enter a password when prompted (remember it!)
3. Reboot: `sudo reboot`
4. During boot, **MOK Manager** will appear (blue screen)
5. Select **"Enroll MOK"**
6. Select **"Continue"**
7. Enter the password you set
8. Select **"Reboot"**
9. After reboot, run: `sudo ./sign_nvidia_module.sh` again
10. The script will sign the modules and load them

## Quick Diagnostic

Run the diagnostic script:
```bash
./fix_nvidia_driver.sh
```

This will tell you exactly what's wrong and which option to choose.

## Verification Steps

After applying either fix:

### 1. Check if modules are loaded:
```bash
lsmod | grep nvidia
```
Should show multiple nvidia modules.

### 2. Test nvidia-smi:
```bash
nvidia-smi
```
Should display:
- GPU: NVIDIA GeForce RTX 3060 Mobile
- Driver Version: 550.163.01
- CUDA Version: 12.0

### 3. Test in Python:
```bash
source venv_gpu_linux/bin/activate
python -c "import cv2; print('CUDA devices:', cv2.cuda.getCudaEnabledDeviceCount())"
```

### 4. Check CUDA:
```bash
nvcc --version
```

## Troubleshooting

### After disabling Secure Boot, nvidia-smi still fails:
```bash
# Reboot first
sudo reboot

# If still failing after reboot:
sudo modprobe nvidia
sudo nvidia-smi
```

### Check kernel logs:
```bash
sudo dmesg | grep nvidia
```

### Reinstall NVIDIA driver if needed:
```bash
sudo apt-get install --reinstall nvidia-driver-550
sudo reboot
```

### Check DKMS status:
```bash
sudo dkms status
```
Should show: `nvidia/550.163.01, 6.14.0-33-generic, x86_64: installed`

## For Your Specific System

**Hardware**: HP Victus with RTX 3060 Mobile  
**OS**: Ubuntu 24.04 (kernel 6.14.0-33)  
**Current Status**: Secure Boot is blocking the driver

**Recommended Action**: **Disable Secure Boot** (Option 1)

Most users with NVIDIA GPUs disable Secure Boot as it's the simplest solution and doesn't impact security for typical desktop use.

## After Fix: Testing GPU with OpenCV

Once nvidia-smi works:

```bash
# Activate your GPU environment
source venv_gpu_linux/bin/activate

# Test OpenCV CUDA (if built with GPU support)
python -c "
import cv2
print('OpenCV version:', cv2.__version__)
print('CUDA enabled:', cv2.cuda.getCudaEnabledDeviceCount() > 0)
if hasattr(cv2, 'cuda'):
    print('CUDA devices:', cv2.cuda.getCudaEnabledDeviceCount())
"

# If you haven't built OpenCV with GPU support yet, run:
# ./install_opencv_gpu.sh
```

## Summary

1. ✓ Your NVIDIA hardware is fine
2. ✓ Driver is installed correctly
3. ✓ CUDA toolkit is installed
4. ✗ Secure Boot is blocking it

**Next step**: Choose Option 1 (disable Secure Boot) or Option 2 (sign modules)

**Easiest path**: Disable Secure Boot in BIOS → reboot → run `nvidia-smi`
