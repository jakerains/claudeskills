# Image Playground Reference

## Scope

Use this reference when integrating Apple's on-device image generation features.

## API Surface and Availability Snapshot

- `ImagePlayground` framework availability: iOS 18.1+, iPadOS 18.1+, macOS 15.1+, Mac Catalyst 18.1+, visionOS 2.4+.
- `ImageCreator` availability: iOS 18.4+, iPadOS 18.4+, macOS 15.4+, Mac Catalyst 18.4+, visionOS 2.4+.
- Images are generated on device.

## Integration Modes

1. SwiftUI sheet APIs (`imagePlaygroundSheet`) for quick system UI integration.
2. `ImagePlaygroundViewController` for UIKit/AppKit control with delegate callbacks.
3. `ImageCreator` for fully programmatic generation via async sequences.

## Availability Check Pattern

```swift
guard ImagePlaygroundViewController.isAvailable else {
    // Hide or disable image generation feature.
    return
}
```

## SwiftUI Sheet Pattern

Use concept text or `[ImagePlaygroundConcept]` and receive a generated image URL on completion.

```swift
.imagePlaygroundSheet(
    isPresented: $isPresented,
    concepts: [.text("A cozy cafe in the rain")],
    sourceImage: nil,
    onCompletion: { url in
        // Load and persist image from URL.
    },
    onCancellation: {
        // Handle cancellation.
    }
)
```

## UIKit/AppKit Pattern

- Configure `ImagePlaygroundViewController` with:
- `concepts`
- optional `sourceImage`
- `selectedGenerationStyle` and `allowedGenerationStyles`
- Implement delegate callbacks for completion and cancellation.

## Programmatic Pattern with ImageCreator

```swift
let creator = try await ImageCreator()
let concepts: [ImagePlaygroundConcept] = [.text("Minimal poster of a mountain skyline")]

for try await generated in creator.images(
    for: concepts,
    style: .illustration,
    limit: 2
) {
    // Handle generated.url and metadata.
}
```

## Styles and Content Inputs

- Core style constants include `.animation`, `.illustration`, and `.sketch`.
- `availableStyles` can be queried at runtime.
- Support optional source images to guide generation where appropriate.
- Keep concept text concise and specific for better results.

## Error and UX Guidance

- Handle `ImageCreator.Error` cases and cancellation cleanly.
- Offer fallback UI when unavailable or unsupported by language/device.
- Keep generation work on the main user flow lightweight; persist outputs quickly.

## Notes on Extended Styles

Apple Intelligence updates mention additional style options in some surfaces (including ChatGPT-powered styles). Verify current behavior per OS release before hard-coding assumptions.

## Sources

- https://developer.apple.com/documentation/imageplayground
- https://developer.apple.com/documentation/imageplayground/imageplaygroundviewcontroller
- https://developer.apple.com/documentation/imageplayground/imagecreator
- https://developer.apple.com/documentation/imageplayground/imageplaygroundstyle
- https://developer.apple.com/apple-intelligence/get-started/
- https://developer.apple.com/apple-intelligence/whats-new/
