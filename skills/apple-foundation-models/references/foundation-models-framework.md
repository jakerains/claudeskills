# Foundation Models Framework Reference

## Scope

Use this reference when building text features with the Foundation Models framework in iOS and macOS apps.

## Platform and Availability Snapshot

- `FoundationModels` framework availability: iOS 26.0+, iPadOS 26.0+, macOS 26.0+, Mac Catalyst 26.0+, visionOS 26.0+.
- `SystemLanguageModel` is the on-device language model in app code.
- Model usage depends on Apple Intelligence device eligibility and whether Apple Intelligence is enabled.

## Local-First Baseline Pattern

```swift
import FoundationModels

let model = SystemLanguageModel.default
guard model.isAvailable else {
    // Fall back to non-AI UX.
    return
}

let session = LanguageModelSession(model: model)
let response = try await session.respond(to: "Summarize this note in 3 bullets.")
```

## Availability-Aware UX Pattern

Use `availability` for targeted UX states.

```swift
switch SystemLanguageModel.default.availability {
case .available:
    // Show AI feature.
case .unavailable(.deviceNotEligible):
    // Device does not support Apple Intelligence.
case .unavailable(.appleIntelligenceNotEnabled):
    // Ask user to enable Apple Intelligence.
case .unavailable(.modelNotReady):
    // Model may be downloading or preparing.
case .unavailable:
    // Generic fallback.
}
```

## Session and Prompting Rules

- Use a new `LanguageModelSession` for one-shot tasks.
- Reuse the same session for multi-turn tasks that need conversation memory.
- Send one request at a time per session. Check `isResponding` before issuing another request.
- Keep prompts task-specific and concise.

## Token and Context Window Constraints

- Session context window is 4,096 tokens total (instructions + prompts + outputs).
- Catch `LanguageModelSession.GenerationError.exceededContextWindowSize(_)`.
- For large inputs, chunk data and process across multiple sessions.

## Guided Generation and Tools

- Use `@Generable` types for typed outputs.
- Use `Tool` to access app logic, local data, or external services.
- Keep tool argument descriptions short; long descriptions increase token usage and latency.
- Inspect `session.transcript` to debug tool-call graphs.

## Locale, Safety, and Guardrails

- Check locale support with `SystemLanguageModel.supportsLocale(_:)`.
- Handle `unsupportedLanguageOrLocale` generation errors.
- Remember that built-in guardrails are scoped to supported languages and locales.

## Performance Tuning

- Call `prewarm(promptPrefix:)` before expected generation to reduce time-to-first-token.
- Reduce token usage with concise instructions and prompts.
- Use Instruments and Foundation Models profiling guidance to measure improvements.
- For guided generation streams, tune `includeSchemaInPrompt` to balance quality versus token cost.

## Adapter Guidance

- Use adapters only for advanced specialization after prompt and tool optimization.
- Adapter files are large (160MB+). Deliver via Background Assets or server-hosted packs.
- Adapters are version-specific to base model versions and require retraining as model versions update.
- Deployment requires `com.apple.developer.foundation-model-adapter` entitlement.

## Sources

- https://developer.apple.com/documentation/foundationmodels
- https://developer.apple.com/documentation/foundationmodels/generating-content-and-performing-tasks-with-foundation-models
- https://developer.apple.com/documentation/foundationmodels/systemlanguagemodel
- https://developer.apple.com/documentation/foundationmodels/expanding-generation-with-tool-calling
- https://developer.apple.com/documentation/foundationmodels/supporting-languages-and-locales-with-foundation-models
- https://developer.apple.com/documentation/foundationmodels/analyzing-the-runtime-performance-of-your-foundation-models-app
- https://developer.apple.com/documentation/foundationmodels/loading-and-using-a-custom-adapter-with-foundation-models
