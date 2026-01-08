# Code Style and Conventions

## General Swift Style
- Standard Swift naming conventions (camelCase for variables/methods, PascalCase for types)
- No explicit style enforcer (SwiftLint/SwiftFormat not configured)
- Follow existing code patterns in the project

## Code Organization

### MARK Comments
Use `// MARK: -` to organize code sections:
```swift
// MARK: - Public API
// MARK: - Private Helpers
// MARK: - Audio System Setup
```

### File Structure
1. Imports
2. Type declaration
3. Properties (published, then private)
4. Initializers
5. Deinit (if needed)
6. Public methods
7. Private methods

## SwiftUI Patterns
- Use `@Published` properties for observable state
- Classes inherit from `ObservableObject` for UI binding
- Views are in separate `Views/` directory

## Async/Concurrency
- Use Swift async/await for asynchronous operations
- Use `@MainActor` for UI-bound code
- Use `Task` for launching async work

## Naming Conventions
- Boolean properties: use `is` prefix (e.g., `isGenerating`, `isAudioPlaying`)
- Factory methods: `make(...)` for creating instances
- Private properties: no underscore prefix (except `_promptURLs` for backing storage)

## Error Handling
- Custom error enums with `LocalizedError` conformance
- Descriptive error messages via `errorDescription` property

## Documentation
- Public API should have documentation comments
- Use `/// ` style for doc comments
- Internal code can use `//` for explanatory comments

## No-Go Patterns (from CLAUDE.md)
- **NO `any` type** - explicit types required
- **NO obvious/useless comments**
- **NO emoji in code** unless explicitly requested
- **FOLLOW KISS principle** - keep it simple
- **FOLLOW DRY principle** - don't repeat yourself
