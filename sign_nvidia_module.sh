#!/bin/bash
# Sign NVIDIA kernel modules for Secure Boot
# This allows the NVIDIA driver to work with Secure Boot enabled

set -e

echo "========================================="
echo "NVIDIA Module Signing for Secure Boot"
echo "========================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "This script must be run as root (use sudo)"
    exit 1
fi

# Generate MOK signing key if it doesn't exist
KEY_DIR="/var/lib/shim-signed/mok"
mkdir -p "$KEY_DIR"

if [ ! -f "$KEY_DIR/MOK.priv" ]; then
    echo "Generating Machine Owner Key (MOK)..."
    openssl req -new -x509 -newkey rsa:2048 -keyout "$KEY_DIR/MOK.priv" \
        -outform DER -out "$KEY_DIR/MOK.der" -nodes -days 36500 \
        -subj "/CN=NVIDIA Driver Signing/"
    
    echo ""
    echo "Enrolling the key with MOK..."
    echo "You will be prompted to set a password."
    echo "Remember this password - you'll need it on next reboot!"
    mokutil --import "$KEY_DIR/MOK.der"
    
    echo ""
    echo "========================================="
    echo "IMPORTANT: REBOOT REQUIRED"
    echo "========================================="
    echo "1. Reboot your computer: sudo reboot"
    echo "2. MOK Manager will appear during boot"
    echo "3. Select 'Enroll MOK'"
    echo "4. Enter the password you just set"
    echo "5. Reboot to complete enrollment"
    echo "6. Run this script again to sign the modules"
    echo "========================================="
    exit 0
fi

# Sign the NVIDIA kernel modules
echo "Signing NVIDIA kernel modules..."
KERNEL_VERSION=$(uname -r)
MODULE_DIR="/lib/modules/$KERNEL_VERSION/updates/dkms"

if [ ! -d "$MODULE_DIR" ]; then
    echo "ERROR: NVIDIA module directory not found: $MODULE_DIR"
    exit 1
fi

# Find and sign all nvidia modules
for module in $(find "$MODULE_DIR" -name "nvidia*.ko"); do
    echo "Signing: $module"
    /usr/src/linux-headers-$KERNEL_VERSION/scripts/sign-file sha256 \
        "$KEY_DIR/MOK.priv" "$KEY_DIR/MOK.der" "$module"
done

echo ""
echo "✓ All NVIDIA modules signed successfully!"
echo ""
echo "Loading NVIDIA modules..."
modprobe nvidia

if lsmod | grep -q nvidia; then
    echo "✓ NVIDIA modules loaded successfully!"
    echo ""
    echo "Testing nvidia-smi..."
    nvidia-smi
else
    echo "ERROR: Failed to load NVIDIA modules"
    echo "Check dmesg for errors: sudo dmesg | tail -50"
    exit 1
fi

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo "NVIDIA driver is now working with Secure Boot enabled."
