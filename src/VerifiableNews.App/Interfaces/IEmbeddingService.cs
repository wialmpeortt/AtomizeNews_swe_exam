namespace VerifiableNews.App.Interfaces;

public interface IEmbeddingService
{
    float[] GenerateEmbedding(string text);
}
