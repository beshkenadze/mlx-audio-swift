# Suggested Commands

## Building

### Via Swift Package Manager
```bash
# Build the library
swift build

# Build for release
swift build -c release
```

### Via Xcode
```bash
# Open macOS app project
open MLXAudio.xcodeproj
# or
open Swift-TTS.xcodeproj

# Build from command line
xcodebuild -project MLXAudio.xcodeproj -scheme MLXAudio build
```

## Testing

```bash
# Run all tests via SPM
swift test

# Run specific test
swift test --filter KokoroTokenizerTests

# Run tests via Xcode
xcodebuild test -project MLXAudio.xcodeproj -scheme MLXAudio
```

## Package Management

```bash
# Update dependencies
swift package update

# Resolve dependencies
swift package resolve

# Show dependency graph
swift package show-dependencies
```

## Cleaning

```bash
# Clean SPM build
swift package clean

# Clean Xcode derived data
rm -rf ~/Library/Developer/Xcode/DerivedData/MLXAudio-*
```

## System Commands (Darwin/macOS)

```bash
# List files
ls -la

# Find files
find . -name "*.swift" -type f

# Search in files
grep -r "pattern" --include="*.swift"

# Git operations
git status
git diff
git log --oneline -10
```

## Notes
- **Do NOT run dev servers** - this is a desktop/mobile app, not a server
- Models download automatically on first use (Marvis)
- Kokoro requires model files to be manually placed in Resources folder
- eSpeak NG framework is already embedded for Kokoro
