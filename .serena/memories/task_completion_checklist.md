# Task Completion Checklist

When completing a coding task in this project, verify the following:

## Before Committing

### 1. Code Compiles
```bash
swift build
```

### 2. Tests Pass
```bash
swift test
```

### 3. Code Style
- [ ] Follows existing code patterns
- [ ] No `any` type usage
- [ ] No obvious/useless comments
- [ ] MARK comments for code sections (if applicable)
- [ ] Proper error handling

### 4. SwiftUI (if applicable)
- [ ] `@Published` properties on main thread
- [ ] ObservableObject pattern followed
- [ ] No retain cycles (weak self in closures)

### 5. Audio Code (if applicable)
- [ ] Proper audio session configuration
- [ ] Resources cleaned up in deinit
- [ ] Thread-safe buffer management

## What NOT to Do
- Do NOT run dev servers
- Do NOT edit package.json dependencies directly (use package manager)
- Do NOT add Co-Authored-By or Generated with Claude in commits
- Do NOT over-engineer - atomic changes only

## Git Commit
- Meaningful commit message
- No auto-generated footer text
