//
//  Qwen3.swift
//  MLXAudio
//
//  Created by Prince Canuma on 29/12/2025.
//

import Foundation
import MLX
import HuggingFace
import Tokenizers
import MLXLMCommon
import MLXFast
import MLXNN
import MLXAudioCodecs
import Combine


// MARK: - VyvoTTS special token IDs (Qwen3-based tokenizer)
let tokenizerLength = 151669
let startOfText = 151643
let endOfText = 151645
let startOfSpeech = tokenizerLength + 1  // 151670
let endOfSpeech = tokenizerLength + 2  // 151671
let startOfHuman = tokenizerLength + 3  // 151672
let endOfHuman = tokenizerLength + 4  // 151673
let startOfAI = tokenizerLength + 5  // 151674
let endOfAI = tokenizerLength + 6  // 151675
let padTokenId = tokenizerLength + 7  // 151676
let audioTokensStart = tokenizerLength + 10  // 151679


// MARK: - Decode
func decodeAudioFromCodes(codeList: [Int], snacModel: SNAC) -> MLXArray {
    var layer1: [Int] = []
    var layer2: [Int] = []
    var layer3: [Int] = []

    let numGroups = (codeList.count + 1) / 7

    for i in 0..<numGroups {
        let baseIdx = 7 * i

        layer1.append(codeList[baseIdx])
        layer2.append(codeList[baseIdx + 1] - 4096)
        layer3.append(codeList[baseIdx + 2] - (2 * 4096))
        layer3.append(codeList[baseIdx + 3] - (3 * 4096))
        layer2.append(codeList[baseIdx + 4] - (4 * 4096))
        layer3.append(codeList[baseIdx + 5] - (5 * 4096))
        layer3.append(codeList[baseIdx + 6] - (6 * 4096))
    }

    let codes = [
        MLXArray(layer1).expandedDimensions(axis: 0),
        MLXArray(layer2).expandedDimensions(axis: 0),
        MLXArray(layer3).expandedDimensions(axis: 0)
    ]

    let audioHat = snacModel.decode(codes).squeezed(axis: -1)
    return audioHat
}

func encodeAudioToCodes(audio: MLXArray, snacModel: SNAC) -> MLXArray {
    // Add batch and channel dimensions: [samples] -> [1, 1, samples]
    let audioExpanded = audio
        .expandedDimensions(axis: 0)
        .expandedDimensions(axis: 0)

    let codes = snacModel.encode(audioExpanded)

    let layer1 = codes[0].squeezed(axis: 0).asArray(Int.self)
    let layer2 = codes[1].squeezed(axis: 0).asArray(Int.self)
    let layer3 = codes[2].squeezed(axis: 0).asArray(Int.self)

    var codeList: [Int] = []
    let numGroups = layer1.count

    for i in 0..<numGroups {
        codeList.append(layer1[i])
        codeList.append(layer2[2 * i] + 4096)
        codeList.append(layer3[4 * i] + 2 * 4096)
        codeList.append(layer3[4 * i + 1] + 3 * 4096)
        codeList.append(layer2[2 * i + 1] + 4 * 4096)
        codeList.append(layer3[4 * i + 2] + 5 * 4096)
        codeList.append(layer3[4 * i + 3] + 6 * 4096)
    }

    return MLXArray(codeList).expandedDimensions(axis: 0)
}

// MARK: - Attention

public class Attention: Module {
    let args: Qwen3Configuration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPE

    public init(_ args: Qwen3Configuration) {
        self.args = args

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads

        let headDim = args.headDim
        self.scale = pow(Float(headDim), -0.5)

        self._wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
        self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        self._wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        let ropeScale: Float
        if let ropeScaling = args.ropeScaling, ropeScaling["type"] == .string("linear"),
        let factor = ropeScaling["factor"]
        {
            if let v = factor.asFloat() {
                ropeScale = 1 / v
            } else {
                fatalError("ropeScaling.factor must be a Float")
            }
        } else {
            ropeScale = 1
        }

        self.rope = RoPE(
            dimensions: headDim, traditional: false, base: args.ropeTheta, scale: ropeScale
        )


    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = qNorm(queries.reshaped(B, L, args.attentionHeads, -1)).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, args.kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)


        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        ).transposed(0, 2, 1, 3).reshaped(B, L, -1)

        return wo(
            output
        )
    }

}


// MARK: - MLP

private class MLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    public init(dimensions: Int, hiddenDimensions: Int) {
        self._gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        self._up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return down(silu(gate(x)) * up(x))
    }
}


private class TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Attention
    let mlp: MLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    public init(_ args: Qwen3Configuration) {
        self._attention.wrappedValue = Attention(args)
        self.mlp = MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        return h + r
    }


}


private class Qwen3ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [TransformerBlock]
    let norm: RMSNorm

    public init(_ args: Qwen3Configuration) {
        precondition(args.vocabularySize > 0)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize,
            dimensions: args.hiddenSize
        )

        self.layers = (0..<args.hiddenLayers)
            .map { _ in TransformerBlock(args) }

        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}


public class Qwen3Model: Module, KVCacheDimensionProvider {

    public let vocabularySize: Int
    public let kvHeads: [Int]
    public var tokenizer: Tokenizer?
    public var _snacModel: SNAC?

    private let model: Qwen3ModelInner

    let configuration: Qwen3Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    // KVCacheDimensionProvider conformance
    public var numLayers: Int {
        return self.configuration.hiddenLayers
    }

    public init(_ args: Qwen3Configuration){
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0..<args.hiddenLayers).map {_ in args.kvHeads}
        self.model = Qwen3ModelInner(args)

        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func parseOutput(_ inputIds: MLXArray) -> [[Int]] {
        let tokenToFind = startOfSpeech
        let tokenToRemove = endOfSpeech

        // Create mask and find indices manually
        let mask = inputIds .== tokenToFind
        var indices: [(Int, Int)] = []

        for i in 0..<mask.shape[0] {
            for j in 0..<mask.shape[1] {
                if mask[i, j].item(Int.self) != 0 {
                    indices.append((i, j))
                }
            }
        }

        var croppedTensor: MLXArray

        // Check if we found any tokens
        if !indices.isEmpty {
            let lastOccurrenceIdx = indices.last!.1
            croppedTensor = inputIds[0..., (lastOccurrenceIdx + 1)...]
        } else {
            croppedTensor = inputIds
        }

        // Process each row
        var processedRows: [MLXArray] = []

        for i in 0..<croppedTensor.shape[0] {
            let row = croppedTensor[i]
            let rowList = row.asArray(Int.self)

            // Filter out tokens to remove
            let maskedRow = rowList.filter { $0 != tokenToRemove }
            processedRows.append(MLXArray(maskedRow))
        }

        // Create code lists
        var codeLists: [[Int]] = []

        for row in processedRows {
            let rowLength = row.shape[0]
            let newLength = (rowLength / 7) * 7
            let trimmedRow = row[0..<newLength]

            // Subtract AUDIO_TOKENS_START from each token
            let trimmedList = trimmedRow.asArray(Int.self)
            let codeList = trimmedList.map { $0 - audioTokensStart }
            codeLists.append(codeList)
        }

        return codeLists
    }

    public func prepareInputIds(
        prompts: [String],
        voice: String? = nil,
        refAudio: MLXArray? = nil,
        refText: String? = nil
    ) -> (MLXArray, MLXArray) {

        var audioInputIds: MLXArray?
        var audioTranscriptIds: MLXArray?

        // Handle reference audio and text
        if let refAudio = refAudio, let refText = refText {
            print("\u{001B}[93mWARNING: Audio cloning doesn't work reliably on this model.\u{001B}[0m")

            guard let snacModel = self._snacModel else {
                fatalError("SNAC model not loaded. Call post_load_hook first.")
            }

            let codes = encodeAudioToCodes(audio: refAudio, snacModel: snacModel)
            audioInputIds = codes + audioTokensStart
            let encodedIds = tokenizer!.encode(text: refText)
            audioTranscriptIds = MLXArray(encodedIds.map { Int32($0) }).expandedDimensions(axis: 0)
        }

        // Apply voice prefix if provided
        var modifiedPrompts = prompts
        if let voice = voice {
            modifiedPrompts = prompts.map { "\(voice): \($0)" }
        }

        // Define special tokens
        let startToken = MLXArray([Int32(startOfHuman)]).expandedDimensions(axis: 0)
        let endTokens = MLXArray([Int32(endOfText), Int32(endOfHuman)]).expandedDimensions(axis: 0)

        // Encode all prompts
        var promptInputIds: [MLXArray] = []
        for prompt in modifiedPrompts {
            let encodedIds = tokenizer!.encode(text: prompt)
            let encoded = MLXArray(encodedIds.map { Int32($0) }).expandedDimensions(axis: 0)
            promptInputIds.append(encoded)
        }

        // Prepare batch with padding
        var batchInputIds: [MLXArray] = []
        let padToken = MLXArray([Int32(padTokenId)])
        let maxLen = promptInputIds.map { $0.shape[1] }.max() ?? 0

        for inputIds in promptInputIds {
            var modifiedInputIds: [MLXArray] = []

            // Add padding if needed
            let paddingLen = maxLen - inputIds.shape[1]
            if paddingLen > 0 {
                let padding = repeated(padToken, count: paddingLen, axis: 0)
                    .expandedDimensions(axis: 0)
                modifiedInputIds.append(padding)
            }

            // Add reference audio and transcript if provided
            if let audioInputIds = audioInputIds, let audioTranscriptIds = audioTranscriptIds {
                let audioStartTokens = MLXArray([
                    Int32(startOfAI), Int32(startOfSpeech)
                ]).expandedDimensions(axis: 0)

                let audioEndTokens = MLXArray([
                    Int32(endOfSpeech), Int32(endOfAI)
                ]).expandedDimensions(axis: 0)

                let refInputIds = concatenated([
                    startToken,
                    audioTranscriptIds,
                    endTokens,
                    audioStartTokens,
                    audioInputIds,
                    audioEndTokens
                ], axis: 1)

                modifiedInputIds.append(refInputIds)
            }

            // Add prompt with start/end tokens
            let onePromptInputIds = concatenated([
                startToken,
                inputIds,
                endTokens
            ], axis: 1)

            modifiedInputIds.append(onePromptInputIds)

            // Concatenate all parts for this prompt
            let fullInputIds = concatenated(modifiedInputIds, axis: 1)
            batchInputIds.append(fullInputIds)
        }

        // Concatenate all prompts in batch
        let finalBatchInputIds = concatenated(batchInputIds, axis: 0)

        // Create attention mask (False for pad tokens, True otherwise)
        let batchMask = finalBatchInputIds .!= padToken

        return (finalBatchInputIds, batchMask)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var out = model(inputs, cache: cache)

        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }

        return out
    }

    public var sampleRate: Int {
        return self.configuration.sampleRate
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights
        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }

        return weights
    }

    public func post_load_hook(model: Qwen3Model, modelDir: URL) async throws {
        if model.tokenizer == nil {
            model.tokenizer = try await AutoTokenizer.from(pretrained: modelDir.path)
        }
        if model._snacModel == nil {
            model._snacModel = try await SNAC.fromPretrained("mlx-community/snac_24khz")
        }
    }

    public static func fromPretrained(_ modelRepo: String) async throws -> Qwen3Model {
        let client = HubClient.default

        let snapshotDir = FileManager.default.temporaryDirectory


        let progress = Progress(totalUnitCount: 0)

        Task {
            for await value in progress.publisher(for: \.fractionCompleted).values {
                print("Snapshot download progress: \(value * 100)%")
            }
        }

        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw NSError(domain: "Qwen3Model", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid repository ID: \(modelRepo)"])
        }

        let modelDir = try await client.downloadSnapshot(
            of: repoID,
            kind: .model,
            to: snapshotDir,
            revision: "main",
            progressHandler: { progress in
            // Accurate progress per file
            print("\(progress.completedUnitCount)/\(progress.totalUnitCount) files")
        })


        let configPath = snapshotDir.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configPath)
        let config = try JSONDecoder().decode(Qwen3Configuration.self, from: configData)

        let quantization = config.quantization
        let perLayerQuantization = config.perLayerQuantization

        let model = Qwen3Model(config)


        // Load weights from safetensors
        let weights = try loadWeights(from: modelDir)

        let sanitizedWeights = model.sanitize(weights: weights)

        // Quantize if needed

        if quantization != nil {
            print("Applying quantizaiton from config...")
            if let quant = quantization {
                print(" Default: groupSize=\(quant.groupSize), bits=\(quant.bits)")
            }
            if let perLayerQuant = perLayerQuantization {
                print(" Per-layer: \(perLayerQuant)")
            }

            quantize(model: model) { path, module in
                // Only quantize if scales exist for this layer
                if weights["\(path).scales"] != nil {
                    if perLayerQuantization != nil {
                        return perLayerQuantization?.quantization(layer: path)?.asTuple
                    } else {
                        return quantization?.asTuple
                    }
                } else {
                    return nil
                }
            }
        }


        try model.update(parameters: ModuleParameters.unflattened(sanitizedWeights), verify: [.all])
        eval(model)

        try await model.post_load_hook(model: model, modelDir: modelDir)

        return model
    }
}

func loadWeights(from directory: URL) throws -> [String: MLXArray] {
    let fileManager = FileManager.default
    let files = try fileManager.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
    let safetensorFiles = files.filter { $0.pathExtension == "safetensors" }

    var weights: [String: MLXArray] = [:]
    for file in safetensorFiles {
        let fileWeights = try MLX.loadArrays(url: file)
        weights.merge(fileWeights) { _, new in new }
    }
    return weights
}
