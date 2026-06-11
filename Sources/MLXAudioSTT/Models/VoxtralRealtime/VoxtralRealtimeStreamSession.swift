import Foundation
import MLX
import MLXAudioCore
import MLXNN

// True incremental (online) streaming for Voxtral Realtime.
//
// The offline `generate(...)` path encodes the entire audio buffer up front and only
// then walks the decoder. This session ingests audio *as it arrives* (e.g. 80 ms mic
// chunks), feeds only newly-frozen conv frames through the transformer encoder with a
// persistent per-layer KV-cache, maintains the decoder KV-cache, and emits tokens with
// the model's native transcription delay — O(1) work per chunk.
//
// Correctness (WER 0 vs offline):
//   * conv stem is causal and `prepareMel` right-pads with zeros, so `convOut[0..<k]`
//     re-derived from a prefix is bit-identical to the offline full encode for every
//     frozen row k. The only unfrozen row is the trailing partial token (chunk ended
//     mid-1280-sample-token) — guarded by `frozenGuardTokens`.
//   * RoPE attention is relative-position invariant, so feeding conv frames in
//     sliding-window-aligned blocks with the cache RESET at each boundary reproduces
//     `encodeChunked` (>sw) exactly; a single un-reset block reproduces `encodeFull`
//     (<=sw). See `feedIncremental`.
//   * `finish()` reproduces the offline tail zero-pad ⇒ final transcript == generate().

/// Persistent incremental-encoder state carried across `step` calls.
struct VoxtralRealtimeStreamEncoderState {
    var caches: [VoxtralRealtimeEncoderKVCache?]
    var blockBase = 0   // absolute conv-frame index where the current sw-block began
    var consumed = 0    // conv frames already fed to the transformer

    init(layers: Int) {
        caches = Array(repeating: nil, count: layers)
    }
}

extension VoxtralRealtimeAudioEncoder {
    /// Feed conv frames `[state.consumed, upTo)` through the transformer incrementally,
    /// resetting the per-layer caches at each `slidingWindow` boundary so the result is
    /// bit-identical to offline `encodeFull` (<=sw) / `encodeChunked` (>sw). Returns the
    /// new transformer-normed frames (pre-downsample).
    func feedIncremental(
        _ convOut: MLXArray,
        upTo: Int,
        state: inout VoxtralRealtimeStreamEncoderState
    ) -> MLXArray {
        feedIncremental(block: convOut[state.consumed..<upTo, 0...], state: &state)
    }

    /// Same, but takes just the new conv frames (row 0 == conv frame `state.consumed`),
    /// so callers can produce them without materialising the whole prefix.
    func feedIncremental(
        block: MLXArray,
        state: inout VoxtralRealtimeStreamEncoderState
    ) -> MLXArray {
        let sw = config.slidingWindow
        let upTo = state.consumed + block.shape[0]
        var pieces: [MLXArray] = []
        var offset = 0
        while state.consumed < upTo {
            let blockEnd = state.blockBase + sw
            let end = min(upTo, blockEnd)
            let n = end - state.consumed
            let sub = block[offset..<(offset + n), 0...]
            // Block-relative positions: RoPE is relative, so this matches the absolute
            // positions offline uses within each independent sw-block.
            let relStart = state.consumed - state.blockBase
            pieces.append(encodeIncremental(sub, startPos: relStart, caches: &state.caches))
            offset += n
            state.consumed = end
            if state.consumed == blockEnd {
                state.caches = Array(repeating: nil, count: transformerLayers.count)
                state.blockBase = blockEnd
            }
        }
        if pieces.isEmpty { return block[0..<0, 0...] }
        return pieces.count == 1 ? pieces[0] : MLX.concatenated(pieces, axis: 0)
    }
}

public final class VoxtralRealtimeStreamSession {
    /// Text + token ids decoded by a single `step` / `finish` call.
    public struct Delta {
        public let text: String
        public let tokenIds: [Int]
    }

    private let model: VoxtralRealtimeModel
    private let temperature: Float
    private let maxTokens: Int
    private let transcriptionDelayMs: Int?

    // Only the trailing partial token (chunk ended mid-1280-sample-token) is unfrozen.
    private let frozenGuardTokens = 1

    private var realAudio: [Float] = []
    private var encState: VoxtralRealtimeStreamEncoderState
    private var adapterBuf: MLXArray?
    private var decCache: [VoxtralRealtimeDecoderKVCache?]?
    private var lastLogits: MLXArray?
    private var decPos = 0
    private var promptLength = 0
    private var prefilled = false
    private var done = false

    private var generated: [Int] = []
    private var emittedText = ""

    private lazy var melFilters: MLXArray = model.ensureMelFilters()
    private lazy var hannWindow: MLXArray = {
        // Periodic Hann — must match `computeMelSpectrogram` bit-for-bit.
        let win = model.config.audioEncodingArgs.windowSize
        let n = MLXArray(0..<win).asType(.float32)
        return 0.5 * (1.0 - cos((2.0 * Float.pi * n) / Float(win)))
    }()

    public init(
        model: VoxtralRealtimeModel,
        temperature: Float = 0.0,
        maxTokens: Int = 4096,
        transcriptionDelayMs: Int? = nil
    ) {
        self.model = model
        self.temperature = temperature
        self.maxTokens = maxTokens
        self.transcriptionDelayMs = transcriptionDelayMs
        self.encState = VoxtralRealtimeStreamEncoderState(
            layers: model.encoder.transformerLayers.count
        )
    }

    /// Full transcript decoded so far.
    public var text: String { emittedText }
    /// Token ids decoded so far (EOS stripped).
    public var tokens: [Int] { generated }
    /// Whether the stream has emitted EOS / hit maxTokens.
    public var isFinished: Bool { done }

    /// Ingest a chunk of 16 kHz mono samples; returns the text decoded by this call.
    @discardableResult
    public func step(_ samples: [Float]) -> Delta {
        realAudio.append(contentsOf: samples)
        return advance(final: false)
    }

    @discardableResult
    public func step(_ samples: MLXArray) -> Delta {
        let flat = samples.ndim > 1 ? samples.mean(axis: -1) : samples
        return step(flat.asType(.float32).asArray(Float.self))
    }

    /// Flush the tail: reproduces the offline zero-pad so the final transcript equals
    /// `generate(...)`. Call once after the last `step`.
    @discardableResult
    public func finish() -> Delta {
        advance(final: true)
    }

    private func advance(final: Bool) -> Delta {
        guard !done else { return Delta(text: "", tokenIds: []) }
        guard !realAudio.isEmpty else { return Delta(text: "", tokenIds: []) }

        let ds = model.config.encoderArgs.downsampleFactor

        // Mid-stream chunks derive shapes arithmetically and compute only the NEW
        // conv rows from a bounded sample window (O(1) per chunk). The first chunk
        // (windowed path needs >= 2 conv rows of left context) and the final flush
        // (right zero-pad region must match offline exactly) run the full stem.
        let nAudioTotal: Int
        let convTotal: Int
        var fullConvOut: MLXArray?

        if final || encState.consumed < 2 {
            let (convOut, total, pLen) = model.convStemForAudio(
                audio: MLXArray(realAudio),
                transcriptionDelayMs: transcriptionDelayMs
            )
            fullConvOut = convOut
            nAudioTotal = total
            convTotal = convOut.shape[0]
            promptLength = pLen
        } else {
            let shapes = streamShapes()
            nAudioTotal = shapes.nAudioTotal
            convTotal = shapes.convTotal
            promptLength = shapes.promptLength
        }

        let realRegion = model.config.nLeftPadTokens + model.numAudioTokens(realAudio.count)
        let emitLimit = final ? nAudioTotal : max(0, min(nAudioTotal, realRegion - frozenGuardTokens))
        let convFreeze = min(convTotal, emitLimit * ds)

        if convFreeze > encState.consumed {
            let newEnc: MLXArray
            if let fullConvOut {
                newEnc = model.encoder.feedIncremental(fullConvOut, upTo: convFreeze, state: &encState)
            } else {
                let block = convRowsWindowed(from: encState.consumed, to: convFreeze)
                newEnc = model.encoder.feedIncremental(block: block, state: &encState)
            }
            let rows = model.encoder.downsampleAndProject(newEnc)   // multiple-of-ds ⇒ whole rows
            adapterBuf = adapterBuf == nil ? rows : MLX.concatenated([adapterBuf!, rows], axis: 0)
            freezeEncoderState()
        }

        guard let adapter = adapterBuf else {
            Memory.clearCache()
            return Delta(text: "", tokenIds: [])
        }
        prefillIfNeeded(adapter: adapter)
        let delta = decode(adapter: adapter, upTo: min(emitLimit, adapter.shape[0]))

        Memory.clearCache()
        return delta
    }

    /// Padded-audio geometry derived arithmetically (mirrors `padAudioStreaming` +
    /// `prepareMel` + `convStem` shape math, all multiples of one token's samples,
    /// so the offline parity-drop / downsample-trunc branches never fire).
    private func streamShapes() -> (nAudioTotal: Int, convTotal: Int, promptLength: Int) {
        let cfg = model.config
        let hop = cfg.audioEncodingArgs.hopLength
        let ds = cfg.encoderArgs.downsampleFactor
        let mult = hop * 2 * ds   // raw samples per audio token (160*2*4 = 1280)

        let delayMs = transcriptionDelayMs ?? cfg.transcriptionDelayMs
        let nDelay = model.numDelayTokens(delayMs)
        let nRight = (nDelay + 1) + 10
        let alignPad = (mult - (realAudio.count % mult)) % mult
        let paddedLen = cfg.nLeftPadTokens * mult + realAudio.count + alignPad + nRight * mult

        let convTotal = paddedLen / hop / 2
        return (convTotal / ds, convTotal, 1 + cfg.nLeftPadTokens + nDelay)
    }

    /// Conv-stem rows `[from, to)` recomputed from a bounded raw-sample window —
    /// bit-equal to `convStem(prepareMel(fullAudio))[from..<to]` for `from >= 2`:
    /// conv row f needs mel frames [2f-3, 2f+1]; mel frame m covers samples
    /// [m*hop - win/2, m*hop + win/2) of the left-padded audio (the STFT reflect
    /// pad resolves to zeros because the left pad is >= win/2 zeros, and frozen
    /// frames never reach the right pad — guarded by `frozenGuardTokens`).
    private func convRowsWindowed(from: Int, to: Int) -> MLXArray {
        let a = model.config.audioEncodingArgs
        let hop = a.hopLength
        let win = a.windowSize
        let leftZeros = model.config.nLeftPadTokens * hop * 2 * model.config.encoderArgs.downsampleFactor

        let m0 = 2 * from - 3
        let mCount = 2 * (to - from) + 3
        let s0 = m0 * hop - win / 2
        let total = (mCount - 1) * hop + win
        precondition(s0 + total - leftZeros <= realAudio.count,
                     "windowed conv-stem read past available audio")

        var buf = [Float](repeating: 0, count: total)
        for i in 0..<total {
            let j = s0 + i - leftZeros
            if j >= 0 { buf[i] = realAudio[j] }
        }

        // Mel: identical op chain to `computeMelSpectrogram` (the global last-frame
        // drop never lands in this window; the log-mel floor is a global constant).
        let frames = asStrided(MLXArray(buf), [mCount, win], strides: [hop, 1], offset: 0) * hannWindow
        var mags = MLX.abs(MLXFFT.rfft(frames, axis: -1)).square()
        mags = mags.transposed(1, 0)
        var logSpec = MLX.log10(MLX.maximum(MLX.matmul(melFilters.transposed(1, 0), mags), MLXArray(Float(1e-10))))
        logSpec = MLX.maximum(logSpec, MLXArray(a.globalLogMelMax - 8.0))
        logSpec = (logSpec + MLXArray(Float(4.0))) / MLXArray(Float(4.0))

        // Valid (un-padded) convs over the window: the causal left context is the
        // extra mel frames included above, so outputs align exactly to [from, to).
        var x = logSpec.transposed(1, 0).expandedDimensions(axis: 0)
        x = gelu(model.encoder.convLayers0Conv.conv(x))
        x = gelu(model.encoder.convLayers1Conv.conv(x))
        return x.squeezed(axis: 0)
    }

    /// Materialise the adapter buffer + encoder caches so the lazy graph stays bounded
    /// across chunks (each chunk would otherwise extend one unbroken graph).
    private func freezeEncoderState() {
        var arrays: [MLXArray] = []
        if let adapterBuf { arrays.append(adapterBuf) }
        for cache in encState.caches {
            if let cache { arrays.append(cache.keys); arrays.append(cache.values) }
        }
        if !arrays.isEmpty { MLX.eval(arrays) }
    }

    private func prefillIfNeeded(adapter: MLXArray) {
        guard !prefilled, adapter.shape[0] >= promptLength else { return }

        let nLeft = model.config.nLeftPadTokens
        let nDelay = promptLength - 1 - nLeft
        let promptIds = [model.config.bosTokenId]
            + Array(repeating: model.config.streamingPadTokenId, count: nLeft + nDelay)
        let promptIdsMX = MLXArray(promptIds.map(Int32.init))
        let promptTextEmbeds = model.decoder.embedTokens(promptIdsMX)

        let prefixEmbeds = adapter[0..<promptLength, 0...] + promptTextEmbeds
        let prefill = model.decoder(prefixEmbeds, startPos: 0, cache: nil)
        lastLogits = model.decoder.logits(prefill.0[prefill.0.shape[0] - 1])
        decCache = prefill.1
        decPos = promptLength
        prefilled = true
        MLX.eval(lastLogits!)
    }

    private func decode(adapter: MLXArray, upTo emitLimit: Int) -> Delta {
        guard prefilled, let startLogits = lastLogits else { return Delta(text: "", tokenIds: []) }

        var newIds: [Int] = []
        // Mirrors the offline `generate` loop exactly (append → check → pop trailing
        // EOS) so the streamed token stream is identical at temperature 0. Same
        // pipelined pattern as offline: queue the next step before reading the token.
        var pendingToken = model.sampleLazy(logits: startLogits, temperature: temperature)
        if decPos < emitLimit { MLX.asyncEval(pendingToken) }

        while decPos < emitLimit {
            let tokenEmbed = model.decoder.embedTokens(pendingToken.reshaped([1]))
            let inputEmbed = decPos < adapter.shape[0]
                ? adapter[decPos..<(decPos + 1), 0...] + tokenEmbed
                : tokenEmbed
            let next = model.decoder(inputEmbed, startPos: decPos, cache: decCache)
            let nextLogits = model.decoder.logits(next.0[0])
            let nextToken = model.sampleLazy(logits: nextLogits, temperature: temperature)
            MLX.asyncEval(nextToken)

            let token = pendingToken.item(Int.self)
            generated.append(token)

            if token == model.config.eosTokenId || generated.count > maxTokens {
                done = true
                if generated.last == model.config.eosTokenId { generated.removeLast() }
                break
            }
            newIds.append(token)

            decCache = next.1
            lastLogits = nextLogits
            pendingToken = nextToken
            decPos += 1
        }

        let textSoFar = model.decodeStreaming(generated)
        let delta: String
        if textSoFar.hasPrefix(emittedText) {
            delta = String(textSoFar.dropFirst(emittedText.count))
        } else {
            delta = textSoFar
        }
        emittedText = textSoFar
        return Delta(text: delta, tokenIds: newIds)
    }
}

public extension VoxtralRealtimeModel {
    /// Create an online streaming session. Feed audio with `step(_:)`, then `finish()`.
    func makeStreamSession(
        temperature: Float = 0.0,
        maxTokens: Int = 4096,
        transcriptionDelayMs: Int? = nil
    ) -> VoxtralRealtimeStreamSession {
        VoxtralRealtimeStreamSession(
            model: self,
            temperature: temperature,
            maxTokens: maxTokens,
            transcriptionDelayMs: transcriptionDelayMs
        )
    }
}
