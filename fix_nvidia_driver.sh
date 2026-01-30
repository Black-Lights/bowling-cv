#!/bin/bash
# NVIDIA Driver Fix Script for Secure Boot Issues
# This script helps fix the "Key was rejected by service" error

echo "========================================="
echo "NVIDIA Driver Secure Boot Fix"
echo "========================================="
echo ""

# Check Secure Boot status
echo "Checking Secure Boot status..."
SB_STATE=$(mokutil --sb-state 2>/dev/null)
echo "$SB_STATE"
echo ""

# Check if nvidia module can be loaded
echo "Checking NVIDIA module status..."
if lsmod | grep -q nvidia; then
    echo "✓ NVIDIA modules are already loaded!"
    nvidia-smi
    exit 0
fi

echo "NVIDIA modules are NOT loaded."
echo ""

# Try to load the module to see the error
echo "Attempting to load NVIDIA module..."
sudo modprobe nvidia 2>&1 | tee /tmp/nvidia_load_error.txt

if grep -q "Key was rejected" /tmp/nvidia_load_error.txt; then
    echo ""
    echo "========================================="
    echo "SECURE BOOT ISSUE DETECTED"
    echo "========================================="
    echo ""
    echo "The NVIDIA driver cannot load because Secure Boot is enabled"
    echo "and the kernel module is not signed."
    echo ""
    echo "You have TWO options:"
    echo ""
    echo "OPTION 1: DISABLE SECURE BOOT (Easier, Recommended)"
    echo "----------------------------------------"
    echo "1. Reboot your computer"
    echo "2. Enter BIOS/UEFI (usually press F2, F10, or DEL during boot)"
    echo "3. Find 'Secure Boot' setting (usually under Security tab)"
    echo "4. Disable Secure Boot"
    echo "5. Save and Exit"
    echo "6. Boot back into Linux"
    echo "7. Run: sudo modprobe nvidia"
    echo "8. Verify with: nvidia-smi"
    echo ""
    echo "OPTION 2: SIGN THE NVIDIA MODULE (More complex)"
    echo "----------------------------------------"
    echo "This keeps Secure Boot enabled but requires signing the module."
    echo "Run: sudo ./sign_nvidia_module.sh"
    echo ""
    echo "For most users, OPTION 1 is recommended."
    echo "========================================="
    
elif grep -q "could not insert" /tmp/nvidia_load_error.txt; then
    echo ""
    echo "There's a different issue loading the NVIDIA module."
    echo "Check the error above and run:"
    echo "  sudo dmesg | tail -50"
    echo "to see kernel logs."
else
    echo ""
    echo "✓ NVIDIA module loaded successfully!"
    echo ""
    echo "Testing nvidia-smi..."
    nvidia-smi
fi

echo ""
echo "Current status:"
echo "  Secure Boot: $SB_STATE"
echo "  NVIDIA modules loaded: $(lsmod | grep nvidia | wc -l) modules"
echo "  Kernel version: $(uname -r)"
echo "  NVIDIA driver version: $(dpkg -l | grep nvidia-driver | awk '{print $3}' | head -1)"
echo ""
