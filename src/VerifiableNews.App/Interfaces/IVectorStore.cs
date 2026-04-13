using VerifiableNews.App.Models;

namespace VerifiableNews.App.Interfaces;

public interface IVectorStore
{
    void IndexChunk(DocumentChunk chunk);
    Task<List<DocumentChunk>> SearchAsync(float[] queryVector, int topK = 5);
}
