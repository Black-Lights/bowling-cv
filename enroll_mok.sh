#!/bin/bash
# Complete MOK enrollment process for NVIDIA driver

echo "========================================="
echo "MOK Key Enrollment for NVIDIA Driver"
echo "========================================="
echo ""

# Check if key exists
if [ ! -f "/var/lib/shim-signed/mok/MOK.der" ]; then
    echo "ERROR: MOK key not found. Run sign_nvidia_module.sh first."
    exit 1
fi

echo "MOK signing key found at: /var/lib/shim-signed/mok/MOK.der"
echo ""

# Check if already enrolled
if mokutil --list-enrolled | grep -q "NVIDIA Driver Signing"; then
    echo "✓ MOK key is already enrolled!"
    echo ""
    echo "The key is enrolled but modules still won't load."
    echo "This means Secure Boot signing is too complex for this setup."
    echo ""
    echo "RECOMMENDED: Disable Secure Boot instead (much simpler)"
    echo "See NVIDIA_SECURE_BOOT_FIX.md for instructions."
    exit 1
fi

echo "MOK key is NOT enrolled yet. Starting enrollment process..."
echo ""
echo "You will be asked to set a password. Remember it!"
echo "You'll need this password during the next boot."
echo ""
read -p "Press Enter to continue..."

# Enroll the key
sudo mokutil --import /var/lib/shim-signed/mok/MOK.der

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "MOK KEY ENROLLMENT INITIATED"
    echo "========================================="
    echo ""
    echo "IMPORTANT: Follow these steps carefully!"
    echo ""
    echo "1. Reboot your computer now: sudo reboot"
    echo ""
    echo "2. During boot, a BLUE SCREEN will appear (MOK Manager)"
    echo "   - Select 'Enroll MOK'"
    echo "   - Select 'Continue'"
    echo "   - Select 'Yes' to confirm"
    echo "   - Enter the password you just set"
    echo "   - Select 'Reboot'"
    echo ""
    echo "3. After reboot, run this to verify:"
    echo "   sudo modprobe nvidia"
    echo "   nvidia-smi"
    echo ""
    echo "========================================="
    echo ""
    echo "Ready to reboot? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        sudo reboot
    else
        echo "Run 'sudo reboot' when ready."
    fi
else
    echo "ERROR: Failed to initiate MOK enrollment"
    exit 1
fi
