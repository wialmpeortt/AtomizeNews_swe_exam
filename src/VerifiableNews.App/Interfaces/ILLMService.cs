using VerifiableNews.App.Models;

namespace VerifiableNews.App.Interfaces;

public interface ILLMService
{
    Task<List<AtomicClaim>> ExtractClaimsAsync(string articleText);
}
