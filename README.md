# LocalVectorEngine — Motore di ricerca vettoriale locale per RAG

**Corso:** Software Engineering — H-Farm College
**Autore:** William
**Progetto:** Zucchetti-7 — motore di ricerca vettoriale locale richiesto dai progetti *AI Document Q&A (RAG)* e *BLite Mobile + AI*
**Timeline:** 4 settimane (Agile Sprint)

## Problema

Gli LLM hanno un knowledge cut-off e non conoscono i tuoi documenti. Il pattern **RAG** (Retrieval-Augmented Generation) risolve il problema in due fasi: prima si indicizzano i documenti trasformandoli in vettori di embedding; poi, a ogni domanda, si recuperano i chunk più simili e li si passano all'LLM come contesto di risposta.

Questo repository fornisce **il motore di ricerca vettoriale** — la parte "retrieval" del RAG — come libreria riutilizzabile. È progettato per essere consumato da due progetti separati:

- *AI Document Q&A* — applicazione RAG che interroga documenti PDF/TXT/MD
- *BLite Mobile + AI* — app .NET MAUI che indicizza documenti in locale sullo smartphone

## Architettura

Il repository è organizzato in **una libreria condivisa** + **una demo console** che la esercita end-to-end.

```
src/
├── LocalVectorEngine.Core/          ← libreria pubblica (il "prodotto")
│   ├── Interfaces/
│   │   ├── IChunkingService.cs      ← spezza un documento in chunk
│   │   ├── IEmbeddingService.cs     ← testo → float[] (vettore)
│   │   └── IVectorStore.cs          ← persistenza + ricerca HNSW
│   └── Models/
│       ├── DocumentChunk.cs         ← record immutabile del chunk
│       └── SearchResult.cs          ← chunk + score di similarità
│
└── LocalVectorEngine.Demo/          ← console app che orchestra la pipeline
    └── Program.cs
```

### Contratti (Core)

```csharp
public interface IChunkingService
{
    IEnumerable<DocumentChunk> Chunk(string documentId, string text, string source);
}

public interface IEmbeddingService
{
    Task<float[]> EmbedAsync(string text, CancellationToken ct = default);
}

public interface IVectorStore
{
    Task StoreChunkAsync(DocumentChunk chunk, float[] embedding, CancellationToken ct = default);
    Task<IReadOnlyList<SearchResult>> SearchAsync(float[] queryEmbedding, int topK, CancellationToken ct = default);
}

public record DocumentChunk(string DocumentId, int ChunkIndex, string Text, string Source);
public record SearchResult(DocumentChunk Chunk, float Score);
```

I metodi asincroni accettano `CancellationToken` per rispettare la spec Zucchetti-7 e permettere timeout/abort sia in contesto server che mobile.

## Pipeline RAG

```
INDEXING (una tantum per documento)

  Documento testuale
        │
        ▼
  IChunkingService         ── split ~512 token, overlap ~50
        │
        ▼
  IEmbeddingService        ── testo → float[384]
        │
        ▼
  IVectorStore.Store       ── persist in BLite (HNSW)


QUERY (per ogni domanda utente)

  Domanda
        │
        ▼
  IEmbeddingService        ── domanda → float[384]
        │
        ▼
  IVectorStore.Search      ── HNSW: top-K chunk più simili
        │
        ▼
  SearchResult[] (chunk + score)  ── pronti da dare in pasto a un LLM
```

## Stack tecnico

| Componente | Tecnologia | Note |
|---|---|---|
| Runtime | .NET 10 | |
| Embedding | ONNX Runtime + `all-MiniLM-L6-v2` | 384 dimensioni, locale |
| Vector DB | BLite 4.3.0 | Indice HNSW integrato |
| Diagrammi | PlantUML | Sorgente in `docs/` |

Nessuna chiamata a servizi cloud: tutto il retrieval è on-device.

## Stato implementazione

| Componente | Interfaccia | Implementazione |
|---|---|---|
| Chunking | ✅ `IChunkingService` | ⏳ Issue #21 |
| Embedding | ✅ `IEmbeddingService` | ⏳ Issue #10 — `OnnxEmbeddingService` |
| Vector store | ✅ `IVectorStore` | ⏳ Issue #11 — `BLiteVectorStore` |
| Demo console | — | ⏳ pipeline end-to-end |

## Come eseguire

```bash
# Build
dotnet build

# Esegui la demo console
dotnet run --project src/LocalVectorEngine.Demo
```

Il modello ONNX `all-MiniLM-L6-v2.onnx` va scaricato in `models/` (ignorato da git per peso: 86 MB).

## Struttura repository

```
Exam/
├── src/                          codice
├── tests/                        test (in progresso)
├── docs/                         diagrammi UML (PlantUML + PNG)
├── models/                       modello ONNX (gitignored)
├── LocalVectorEngine.slnx        solution file
└── README.md
```

## Documentazione di progetto

- `docs/class_diagram.puml` — diagramma delle classi (Core + implementazioni previste)
- `docs/sequence_diagram.puml` — sequence dei flussi di indexing e query
- Il Kanban board su GitHub Projects tiene traccia del lavoro sprint-per-sprint.
