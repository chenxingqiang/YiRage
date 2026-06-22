#!/bin/bash
# =============================================================================
# YiRage LLVM/MLIR Installation Script
# =============================================================================
# This script installs LLVM/MLIR with the required components for YiRage.
#
# Supported platforms:
#   - Ubuntu/Debian (apt)
#   - macOS (brew)
#   - CentOS/RHEL/Fedora (dnf/yum)
#   - Windows (MSYS2/chocolatey)
#
# Usage:
#   ./scripts/install_llvm.sh [--version VERSION] [--prefix PATH]
#
# Options:
#   --version VERSION   LLVM version to install (default: 17)
#   --prefix PATH       Installation prefix (default: /usr/local for source builds)
#   --source            Build from source (for custom builds)
#   --help              Show this help message
#
# Environment Variables:
#   LLVM_VERSION        Override LLVM version
#   LLVM_PREFIX         Override installation prefix
#   MLIR_DIR            Will be set after installation
#
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
LLVM_VERSION="${LLVM_VERSION:-17}"
LLVM_PREFIX="${LLVM_PREFIX:-/usr/local}"
BUILD_FROM_SOURCE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --version)
            LLVM_VERSION="$2"
            shift 2
            ;;
        --prefix)
            LLVM_PREFIX="$2"
            shift 2
            ;;
        --source)
            BUILD_FROM_SOURCE=true
            shift
            ;;
        --help)
            head -35 "$0" | tail -31
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/debian_version ]; then
            echo "debian"
        elif [ -f /etc/redhat-release ]; then
            echo "redhat"
        elif [ -f /etc/arch-release ]; then
            echo "arch"
        else
            echo "linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

OS=$(detect_os)
echo -e "${BLUE}Detected OS: ${OS}${NC}"
echo -e "${BLUE}Installing LLVM version: ${LLVM_VERSION}${NC}"

# =============================================================================
# Ubuntu/Debian Installation
# =============================================================================
install_debian() {
    echo -e "${GREEN}Installing LLVM ${LLVM_VERSION} on Debian/Ubuntu...${NC}"
    
    # Add LLVM APT repository
    wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | sudo tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc
    
    # Determine distribution codename
    DISTRO=$(lsb_release -cs)
    
    # Add LLVM repository
    echo "deb http://apt.llvm.org/${DISTRO}/ llvm-toolchain-${DISTRO}-${LLVM_VERSION} main" | \
        sudo tee /etc/apt/sources.list.d/llvm-${LLVM_VERSION}.list
    
    sudo apt-get update
    
    # Install LLVM, Clang, and MLIR
    sudo apt-get install -y \
        llvm-${LLVM_VERSION} \
        llvm-${LLVM_VERSION}-dev \
        llvm-${LLVM_VERSION}-tools \
        clang-${LLVM_VERSION} \
        libclang-${LLVM_VERSION}-dev \
        mlir-${LLVM_VERSION}-tools \
        libmlir-${LLVM_VERSION}-dev
    
    # Create symlinks for default versions
    sudo update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-${LLVM_VERSION} 100
    sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-${LLVM_VERSION} 100
    sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-${LLVM_VERSION} 100
    
    # Set MLIR_DIR for CMake
    MLIR_DIR="/usr/lib/llvm-${LLVM_VERSION}/lib/cmake/mlir"
    LLVM_DIR="/usr/lib/llvm-${LLVM_VERSION}/lib/cmake/llvm"
    
    echo -e "${GREEN}LLVM ${LLVM_VERSION} installed successfully!${NC}"
    echo -e "${YELLOW}Set the following environment variables:${NC}"
    echo "export MLIR_DIR=${MLIR_DIR}"
    echo "export LLVM_DIR=${LLVM_DIR}"
    echo "export PATH=\"/usr/lib/llvm-${LLVM_VERSION}/bin:\$PATH\""
}

# =============================================================================
# macOS Installation (Homebrew)
# =============================================================================
install_macos() {
    echo -e "${GREEN}Installing LLVM ${LLVM_VERSION} on macOS...${NC}"
    
    # Check for Homebrew
    if ! command -v brew &> /dev/null; then
        echo -e "${YELLOW}Homebrew not found. Installing...${NC}"
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    # Install LLVM
    brew install llvm@${LLVM_VERSION}
    
    # Get LLVM prefix
    LLVM_PREFIX=$(brew --prefix llvm@${LLVM_VERSION})
    MLIR_DIR="${LLVM_PREFIX}/lib/cmake/mlir"
    LLVM_DIR="${LLVM_PREFIX}/lib/cmake/llvm"
    
    echo -e "${GREEN}LLVM ${LLVM_VERSION} installed successfully!${NC}"
    echo -e "${YELLOW}Set the following environment variables:${NC}"
    echo "export MLIR_DIR=${MLIR_DIR}"
    echo "export LLVM_DIR=${LLVM_DIR}"
    echo "export PATH=\"${LLVM_PREFIX}/bin:\$PATH\""
    echo "export LDFLAGS=\"-L${LLVM_PREFIX}/lib\""
    echo "export CPPFLAGS=\"-I${LLVM_PREFIX}/include\""
}

# =============================================================================
# CentOS/RHEL/Fedora Installation
# =============================================================================
install_redhat() {
    echo -e "${GREEN}Installing LLVM ${LLVM_VERSION} on CentOS/RHEL/Fedora...${NC}"
    
    # Detect package manager
    if command -v dnf &> /dev/null; then
        PKG_MANAGER="dnf"
    else
        PKG_MANAGER="yum"
    fi
    
    # Enable EPEL and PowerTools for older versions
    if [ -f /etc/centos-release ]; then
        sudo ${PKG_MANAGER} install -y epel-release
        sudo ${PKG_MANAGER} config-manager --set-enabled powertools || true
    fi
    
    # Install LLVM
    sudo ${PKG_MANAGER} install -y \
        llvm${LLVM_VERSION} \
        llvm${LLVM_VERSION}-devel \
        clang${LLVM_VERSION} \
        clang${LLVM_VERSION}-devel \
        mlir${LLVM_VERSION} \
        mlir${LLVM_VERSION}-devel
    
    MLIR_DIR="/usr/lib64/cmake/mlir"
    LLVM_DIR="/usr/lib64/cmake/llvm"
    
    echo -e "${GREEN}LLVM ${LLVM_VERSION} installed successfully!${NC}"
    echo -e "${YELLOW}Set the following environment variables:${NC}"
    echo "export MLIR_DIR=${MLIR_DIR}"
    echo "export LLVM_DIR=${LLVM_DIR}"
}

# =============================================================================
# Arch Linux Installation
# =============================================================================
install_arch() {
    echo -e "${GREEN}Installing LLVM on Arch Linux...${NC}"
    
    sudo pacman -S --noconfirm llvm clang mlir
    
    MLIR_DIR="/usr/lib/cmake/mlir"
    LLVM_DIR="/usr/lib/cmake/llvm"
    
    echo -e "${GREEN}LLVM installed successfully!${NC}"
    echo -e "${YELLOW}Set the following environment variables:${NC}"
    echo "export MLIR_DIR=${MLIR_DIR}"
    echo "export LLVM_DIR=${LLVM_DIR}"
}

# =============================================================================
# Windows Installation (MSYS2/Chocolatey)
# =============================================================================
install_windows() {
    echo -e "${GREEN}Installing LLVM on Windows...${NC}"
    
    if command -v choco &> /dev/null; then
        # Chocolatey installation
        choco install llvm -y --version=${LLVM_VERSION}.0.0
        LLVM_PREFIX="C:/Program Files/LLVM"
    elif command -v pacman &> /dev/null; then
        # MSYS2 installation
        pacman -S --noconfirm mingw-w64-x86_64-llvm mingw-w64-x86_64-clang mingw-w64-x86_64-mlir
        LLVM_PREFIX="/mingw64"
    else
        echo -e "${RED}No supported package manager found. Please install LLVM manually.${NC}"
        exit 1
    fi
    
    MLIR_DIR="${LLVM_PREFIX}/lib/cmake/mlir"
    LLVM_DIR="${LLVM_PREFIX}/lib/cmake/llvm"
    
    echo -e "${GREEN}LLVM installed successfully!${NC}"
    echo -e "${YELLOW}Set the following environment variables:${NC}"
    echo "set MLIR_DIR=${MLIR_DIR}"
    echo "set LLVM_DIR=${LLVM_DIR}"
}

# =============================================================================
# Build from Source
# =============================================================================
build_from_source() {
    echo -e "${GREEN}Building LLVM ${LLVM_VERSION} from source...${NC}"
    
    # Check dependencies
    for cmd in cmake ninja git; do
        if ! command -v $cmd &> /dev/null; then
            echo -e "${RED}${cmd} is required but not installed.${NC}"
            exit 1
        fi
    done
    
    WORKDIR=$(mktemp -d)
    cd ${WORKDIR}
    
    echo -e "${BLUE}Cloning LLVM repository (shallow clone)...${NC}"
    git clone --depth 1 --branch llvmorg-${LLVM_VERSION}.0.0 \
        https://github.com/llvm/llvm-project.git
    
    cd llvm-project
    mkdir build && cd build
    
    echo -e "${BLUE}Configuring LLVM build...${NC}"
    cmake -G Ninja ../llvm \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=${LLVM_PREFIX} \
        -DLLVM_ENABLE_PROJECTS="mlir;clang" \
        -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
        -DLLVM_ENABLE_RTTI=ON \
        -DLLVM_BUILD_EXAMPLES=OFF \
        -DLLVM_INCLUDE_TESTS=OFF \
        -DLLVM_INCLUDE_BENCHMARKS=OFF
    
    echo -e "${BLUE}Building LLVM (this may take a while)...${NC}"
    ninja -j$(nproc)
    
    echo -e "${BLUE}Installing LLVM...${NC}"
    sudo ninja install
    
    # Cleanup
    cd /
    rm -rf ${WORKDIR}
    
    MLIR_DIR="${LLVM_PREFIX}/lib/cmake/mlir"
    LLVM_DIR="${LLVM_PREFIX}/lib/cmake/llvm"
    
    echo -e "${GREEN}LLVM ${LLVM_VERSION} built and installed successfully!${NC}"
    echo -e "${YELLOW}Set the following environment variables:${NC}"
    echo "export MLIR_DIR=${MLIR_DIR}"
    echo "export LLVM_DIR=${LLVM_DIR}"
    echo "export PATH=\"${LLVM_PREFIX}/bin:\$PATH\""
}

# =============================================================================
# Verify Installation
# =============================================================================
verify_installation() {
    echo -e "${BLUE}Verifying LLVM/MLIR installation...${NC}"
    
    # Check llvm-config
    if command -v llvm-config &> /dev/null; then
        echo -e "${GREEN}✓ llvm-config found: $(llvm-config --version)${NC}"
    elif command -v llvm-config-${LLVM_VERSION} &> /dev/null; then
        echo -e "${GREEN}✓ llvm-config-${LLVM_VERSION} found: $(llvm-config-${LLVM_VERSION} --version)${NC}"
    else
        echo -e "${RED}✗ llvm-config not found${NC}"
    fi
    
    # Check mlir-opt
    if command -v mlir-opt &> /dev/null; then
        echo -e "${GREEN}✓ mlir-opt found${NC}"
    elif command -v mlir-opt-${LLVM_VERSION} &> /dev/null; then
        echo -e "${GREEN}✓ mlir-opt-${LLVM_VERSION} found${NC}"
    else
        echo -e "${YELLOW}⚠ mlir-opt not found (may be in non-standard path)${NC}"
    fi
    
    # Check for MLIR CMake config
    if [ -d "${MLIR_DIR}" ]; then
        echo -e "${GREEN}✓ MLIR CMake config found: ${MLIR_DIR}${NC}"
    else
        echo -e "${YELLOW}⚠ MLIR CMake config not found at: ${MLIR_DIR}${NC}"
        echo -e "${YELLOW}  You may need to set MLIR_DIR manually${NC}"
    fi
}

# =============================================================================
# Write environment file
# =============================================================================
write_env_file() {
    local ENV_FILE="${HOME}/.yirage_mlir_env"
    
    cat > ${ENV_FILE} << EOF
# YiRage MLIR Environment Variables
# Source this file: source ${ENV_FILE}

export LLVM_VERSION=${LLVM_VERSION}
export MLIR_DIR=${MLIR_DIR}
export LLVM_DIR=${LLVM_DIR}

# Add to PATH if using custom prefix
if [ -d "${LLVM_PREFIX}/bin" ]; then
    export PATH="${LLVM_PREFIX}/bin:\$PATH"
fi

# For CMake
export CMAKE_PREFIX_PATH="${MLIR_DIR}:\${CMAKE_PREFIX_PATH}"
EOF
    
    echo -e "${GREEN}Environment file written to: ${ENV_FILE}${NC}"
    echo -e "${YELLOW}Run: source ${ENV_FILE}${NC}"
}

# =============================================================================
# Main Installation Logic
# =============================================================================
main() {
    echo -e "${BLUE}=============================================${NC}"
    echo -e "${BLUE}   YiRage LLVM/MLIR Installation Script${NC}"
    echo -e "${BLUE}=============================================${NC}"
    
    if [ "$BUILD_FROM_SOURCE" = true ]; then
        build_from_source
    else
        case $OS in
            debian)
                install_debian
                ;;
            macos)
                install_macos
                ;;
            redhat)
                install_redhat
                ;;
            arch)
                install_arch
                ;;
            windows)
                install_windows
                ;;
            *)
                echo -e "${YELLOW}Unsupported OS. Building from source...${NC}"
                build_from_source
                ;;
        esac
    fi
    
    verify_installation
    write_env_file
    
    echo ""
    echo -e "${GREEN}=============================================${NC}"
    echo -e "${GREEN}   Installation Complete!${NC}"
    echo -e "${GREEN}=============================================${NC}"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "1. Source the environment file:"
    echo "   source ~/.yirage_mlir_env"
    echo ""
    echo "2. Configure YiRage with MLIR support:"
    echo "   cp cmake/backends/mlir.cmake config.cmake"
    echo "   # Or edit config.cmake and set USE_MLIR=ON"
    echo ""
    echo "3. Build YiRage:"
    echo "   mkdir build && cd build"
    echo "   cmake .. -DMLIR_DIR=\${MLIR_DIR}"
    echo "   make -j\$(nproc)"
}

main "$@"
