using LocalVectorEngine.Core.Indexing;
using LocalVectorEngine.Core.Retrieval;
using LocalVectorEngine.Core.Services;

// ═══════════════════════════════════════════════════════════════════════
// LocalVectorEngine.Demo — CLI that exercises the full RAG pipeline
//
//   dotnet run --project src/LocalVectorEngine.Demo -- index <file>
//   dotnet run --project src/LocalVectorEngine.Demo -- query "your question"
//   dotnet run --project src/LocalVectorEngine.Demo -- demo
//
// The "demo" command indexes the sample document shipped in samples/
// and then runs a few pre-canned queries to show the pipeline end-to-end.
// ═══════════════════════════════════════════════════════════════════════

// ── Locate artefacts ────────────────────────────────────────────────────
string repoRoot  = FindRepoRoot(AppContext.BaseDirectory);
string modelPath = Env("LVE_MODEL_PATH") ?? Path.Combine(repoRoot, "models", "all-MiniLM-L6-v2.onnx");
string vocabPath = Env("LVE_VOCAB_PATH") ?? Path.Combine(repoRoot, "models", "vocab.txt");
string dbPath    = Env("LVE_DB_PATH")    ?? Path.Combine(repoRoot, "data", "vectorstore.db");

if (args.Length == 0)
{
    PrintUsage();
    return 1;
}

string command = args[0].ToLowerInvariant();

// ── Pre-flight checks ───────────────────────────────────────────────────
if (!File.Exists(modelPath) || !File.Exists(vocabPath))
{
    Console.Error.WriteLine("ERROR: Missing ONNX model or vocab file.");
    Console.Error.WriteLine($"  model: {modelPath}");
    Console.Error.WriteLine($"  vocab: {vocabPath}");
    Console.Error.WriteLine("See README for download instructions.");
    return 1;
}

// Ensure the database directory exists.
var dbDir = Path.GetDirectoryName(Path.GetFullPath(dbPath));
if (!string.IsNullOrEmpty(dbDir) && !Directory.Exists(dbDir))
    Directory.CreateDirectory(dbDir);

// ── Wire up the services (manual DI) ────────────────────────────────────
using var embedder = new OnnxEmbeddingService(modelPath, vocabPath);
var chunker = new SlidingWindowChunkingService(chunkSizeWords: 100, overlapWords: 25);

// BLiteVectorStore is IDisposable — we create/dispose per command because
// the DB file is a single-writer resource.
switch (command)
{
    case "index":
        return await RunIndex(args);

    case "query":
        return await RunQuery(args);

    case "demo":
        return await RunDemo();

    default:
        Console.Error.WriteLine($"Unknown command: {command}");
        PrintUsage();
        return 1;
}

// ═══════════════════════════════════════════════════════════════════════
// Commands
// ═══════════════════════════════════════════════════════════════════════

async Task<int> RunIndex(string[] a)
{
    if (a.Length < 2)
    {
        Console.Error.WriteLine("Usage: index <file> [file2 …]");
        return 1;
    }

    using var store   = new BLiteVectorStore(dbPath);
    var indexer       = new DocumentIndexer(chunker, embedder, store);
    int totalChunks   = 0;

    for (int i = 1; i < a.Length; i++)
    {
        var filePath = a[i];
        if (!File.Exists(filePath))
        {
            Console.Error.WriteLine($"File not found: {filePath}");
            continue;
        }

        Console.Write($"Indexing {Path.GetFileName(filePath)} … ");
        int n = await indexer.IndexFileAsync(filePath);
        Console.WriteLine($"{n} chunks stored.");
        totalChunks += n;
    }

    Console.WriteLine($"\nDone — {totalChunks} chunks indexed into {dbPath}");
    return 0;
}

async Task<int> RunQuery(string[] a)
{
    if (a.Length < 2)
    {
        Console.Error.WriteLine("Usage: query \"your question here\"");
        return 1;
    }

    string question = string.Join(' ', a.Skip(1));

    using var store = new BLiteVectorStore(dbPath);
    var retriever   = new RetrievalEngine(embedder, store);

    Console.WriteLine($"Question: {question}\n");

    var results = await retriever.RetrieveAsync(question, topK: 5);

    if (results.Count == 0)
    {
        Console.WriteLine("No results. Have you indexed any documents yet?");
        return 0;
    }

    for (int i = 0; i < results.Count; i++)
    {
        var r = results[i];
        Console.WriteLine($"── Result {i + 1}  (score: {r.Score:F4}) ──");
        Console.WriteLine($"   Source: {r.Chunk.Source} [chunk {r.Chunk.ChunkIndex}]");
        Console.WriteLine($"   {Truncate(r.Chunk.Text, 200)}");
        Console.WriteLine();
    }

    return 0;
}

async Task<int> RunDemo()
{
    string sampleDir = Path.Combine(repoRoot, "samples");
    string sampleFile = Path.Combine(sampleDir, "rag_overview.txt");

    if (!File.Exists(sampleFile))
    {
        Console.Error.WriteLine($"Sample file not found: {sampleFile}");
        Console.Error.WriteLine("Make sure the samples/ directory exists in the repo root.");
        return 1;
    }

    // ── Index ──
    Console.WriteLine("╔══════════════════════════════════════════════════╗");
    Console.WriteLine("║  LocalVectorEngine — End-to-End Demo            ║");
    Console.WriteLine("╚══════════════════════════════════════════════════╝\n");

    using var store = new BLiteVectorStore(dbPath);
    var indexer     = new DocumentIndexer(chunker, embedder, store);
    var retriever   = new RetrievalEngine(embedder, store);

    Console.Write($"Indexing {Path.GetFileName(sampleFile)} … ");
    int chunks = await indexer.IndexFileAsync(sampleFile);
    Console.WriteLine($"{chunks} chunks stored.\n");

    // ── Query ──
    string[] questions =
    {
        "What is Retrieval-Augmented Generation?",
        "How does HNSW search work?",
        "What is the embedding dimension of MiniLM?",
    };

    foreach (var q in questions)
    {
        Console.WriteLine($"Q: {q}");
        var results = await retriever.RetrieveAsync(q, topK: 3);

        for (int i = 0; i < results.Count; i++)
        {
            var r = results[i];
            Console.WriteLine($"  [{i + 1}] score={r.Score:F4}  chunk#{r.Chunk.ChunkIndex}  \"{Truncate(r.Chunk.Text, 120)}\"");
        }
        Console.WriteLine();
    }

    Console.WriteLine("Demo complete.");
    return 0;
}

// ═══════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════

void PrintUsage()
{
    Console.WriteLine("LocalVectorEngine.Demo — local RAG pipeline CLI\n");
    Console.WriteLine("Commands:");
    Console.WriteLine("  index <file> [file2 …]     Index one or more text files");
    Console.WriteLine("  query \"your question\"       Retrieve top-K passages");
    Console.WriteLine("  demo                        Index sample doc + run sample queries");
    Console.WriteLine();
    Console.WriteLine("Environment variables:");
    Console.WriteLine("  LVE_MODEL_PATH   Path to all-MiniLM-L6-v2.onnx");
    Console.WriteLine("  LVE_VOCAB_PATH   Path to vocab.txt");
    Console.WriteLine("  LVE_DB_PATH      Path to BLite database file");
}

static string Truncate(string text, int maxLen)
{
    var oneLine = text.ReplaceLineEndings(" ");
    return oneLine.Length <= maxLen ? oneLine : oneLine[..maxLen] + "…";
}

static string? Env(string name) => Environment.GetEnvironmentVariable(name);

static string FindRepoRoot(string start)
{
    var dir = new DirectoryInfo(start);
    while (dir is not null)
    {
        if (dir.GetFiles("*.slnx").Length > 0 || Directory.Exists(Path.Combine(dir.FullName, ".git")))
            return dir.FullName;
        dir = dir.Parent;
    }
    return start;
}
