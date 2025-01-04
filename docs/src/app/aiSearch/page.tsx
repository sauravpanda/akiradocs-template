"use client"

import { useState, useCallback } from 'react'
import { SearchHeader } from '@/components/aiSearch/SearchHeader'
import { SearchBar } from '@/components/aiSearch/SearchBar'
import { LegacyDocsToggle } from '@/components/aiSearch/LegacyDocsToggle'
import { AIResponse } from '@/components/aiSearch/AIResponse'
import { RecommendedArticles } from '@/components/aiSearch/RecommendedArticles'
import { AnimatePresence } from 'framer-motion'
import { getRecommendedArticles } from '@/lib/recommendedArticles'
import { getSearchConfig } from '@/lib/searchConfig'
import { Button } from '@/components/ui/button'
import Link from 'next/link'
import { getAkiradocsConfig } from '@/lib/getAkiradocsConfig'
import { ChatCompletionMessageParam, CreateMLCEngine } from "@mlc-ai/web-llm";
import { Source } from '@/types/Source'
import AILoader from '@/components/aiSearch/AILoader'
import { getHeaderConfig } from '@/lib/headerConfig'
import { Header } from '@/components/layout/Header'
import { generateEmbedding } from '@/lib/aisearch/embeddings'
import { getDbWorker } from '@/lib/aisearch/dbWorker'

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.some(isNaN) || b.some(isNaN)) {
    console.error("NaN values detected in vectors:");
    console.error("Vector A NaN indices:", a.map((val, i) => isNaN(val) ? i : null).filter(x => x !== null));
    console.error("Vector B NaN indices:", b.map((val, i) => isNaN(val) ? i : null).filter(x => x !== null));
    throw new Error("Invalid vectors containing NaN values");
  }

  const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  return dotProduct / (magnitudeA * magnitudeB);
}

export default function Home() {
  const [query, setQuery] = useState('')
  const [aiResponse, setAiResponse] = useState('')
  const [loaderText, setLoaderText] = useState('Loading database ...')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const recommendedArticles = getRecommendedArticles()
  const searchConfig = getSearchConfig()
  const headerConfig = getHeaderConfig()
  const config = getAkiradocsConfig()
  const [sources, setSources] = useState<Source[]>([])
  const handleGenerateEmbedding = useCallback(async (text: string) => {
    try {
      setIsLoading(true);
      // console.log("Loading model for embedding");
      const embedding = await generateEmbedding(text, (progress) => { });
      return embedding;
    } catch (error) {
      console.error('Error generating embedding:', error);
      throw error;
    }
    finally {
      setLoaderText('Searching database for relevant information ...')
    }
  }, []);

  // If AI Search is disabled, show the disabled message
  if (!config.navigation.header.items.find((item: any) => item.href === '/aiSearch')?.show) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center p-4">
        <div className="text-center space-y-4 max-w-md">
          <h1 className="text-3xl font-bold">AI Search is Disabled</h1>
          <p className="text-muted-foreground">
            AI Search is currently disabled. To enable this feature, set &quot;aiSearch&quot;: true in your configuration file.
          </p>
          <Button asChild>
            <Link href="/">Go to Home</Link>
          </Button>
        </div>
      </div>
    )
  }

  const extractSources = (response: string): { cleanResponse: string, sources: Source[] } => {
    // Split the response into parts
    const parts = response.split('Sources:');
    if (parts.length < 2) {
      return { cleanResponse: response, sources: [] };
    }

    const cleanResponse = parts[0].trim();
    const sourcesText = parts[1].trim();

    // Extract sources from the text
    const sources: Source[] = sourcesText
      .split('\n')
      .filter(line => line.trim().length > 0 && line.includes('.json')) // Only process non-empty lines containing .json
      .map(line => ({
        title: line.trim().split('/').pop()?.replace('.json', '') || '', // Get the filename without extension
        url: line.trim()
      }));

    return { cleanResponse, sources };
  };

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    setError(null)
    setSources([])

    const startTime = performance.now()

    try {
      // Generate embedding for the query
      const queryEmbedding = await handleGenerateEmbedding(query);
      // console.log("Query embedding:", queryEmbedding)
      // Get database worker
      const worker = await getDbWorker();

      // Get all documents
      const allDocs = await worker.db.query(`
                SELECT path, content, embedding
                FROM documents
                WHERE embedding IS NOT NULL
            `);

      // Calculate similarity scores and filter results
      const similarityThreshold = 0.5;
      const scoredDocs = allDocs
        .map((doc: any) => {
          // Clean the embedding string and parse it
          const cleanEmbeddingStr = doc.embedding.replace(/[\[\]]/g, ''); // Remove square brackets
          const embeddingArray = cleanEmbeddingStr
            .split(',')
            .map((val: string) => {
              const parsed = parseFloat(val.trim());
              if (isNaN(parsed)) {
                console.error(`Invalid embedding value found: "${val}"`);
              }
              return parsed;
            });

          return {
            ...doc,
            similarity_score: cosineSimilarity(queryEmbedding, embeddingArray)
          };
        })
        .filter((doc: any) => doc.similarity_score > similarityThreshold)
        .sort((a: any, b: any) => b.similarity_score - a.similarity_score)
        .slice(0, 5);

      console.log("RAG top 5 results:", scoredDocs);

      // If no relevant documents found, return early
      if (scoredDocs.length === 0) {
        setAiResponse("I cannot answer this question from the given documentation. The available content doesn't seem relevant enough to provide a accurate answer.");
        setIsLoading(false);
        return;
      }

      setLoaderText('Loading the AI model ...')

      // Combine relevant documents into context
      const docsContext = scoredDocs
        .map((doc: any) => `
                    Source: ${doc.path}
                    --- Content ---
                    ${doc.content}
                    --- End of Content ---
                `)
        .join('\n');

      const engine = await CreateMLCEngine(
        "Llama-3.2-1B-Instruct-q4f16_1-MLC",
        {
          initProgressCallback: (progress: any) => {
            console.log(progress)
            setLoaderText(`Loading the AI model ${Math.round(progress.progress * 100)}% ...`)
          }
        },
        {
          context_window_size: 20000,
        }
      );



      const engineLoadTime = performance.now() // Track engine load time
      console.log(`Time taken for engine initialization: ${(engineLoadTime - startTime) / 1000}s`)
      setLoaderText('Processing information and generating AI response ...')
      const messages = [
        {
          role: "system",
          content: `You are a technical documentation assistant for AkiraDocs. Your purpose is to:
1. Provide accurate, helpful answers using ONLY the provided documentation
2. Stay positive and factual based on the documentation provided.
3. Make sure the markdown answer is pretty, clean and easy to read.`
        },
        {
          role: "user",
          content: `
                  Please provide a helpful answer which is short and concise to the following question using only the provided documentation.

          Question: ${query}

          Answer the question using only the provided documentation. 
                    
          Do not make assumptions or add information not in the documentation.
          
          If relevant, include short code snippets. Only add code snippets if it is helpful to the question.

          Dont add any notes or comments to the answer.

          Make sure the question is answered properly avoiding unnecessary information.

          Ensure the final output is in markdown format. Make sure it pretty and clean.
          
          End of the answer, if you used any sources from the documentation, list them in this format:
          -------------
          Sources:
          - <title> (<path>)
          
          Here is the documentation, only answer based on this information: 
          ${docsContext}
          `
        }
      ];

      // console.log("Messages:", messages)

      const chunks = await engine.chat.completions.create({
        messages: messages as ChatCompletionMessageParam[],
        stream: true,
        stream_options: { include_usage: false },
        max_tokens: 300,
        temperature: 0.5,
        top_p: 0.8,
        frequency_penalty: 0.3,
        presence_penalty: 0.3,
      });

      let aiContent = "";
      for await (const chunk of chunks) {
        const newContent = chunk.choices[0]?.delta.content || "";
        aiContent += newContent;

        // Process partial content for streaming
        const { cleanResponse } = extractSources(aiContent);
        setAiResponse(cleanResponse);
        setIsLoading(false)
      }


      // Only extract and set sources after streaming is complete
      const { sources } = extractSources(aiContent);
      setSources(sources);

      const endTime = performance.now() // Track total time
      console.log(`Total time taken for AI search: ${(endTime - startTime) / 1000}s`)

    } catch (error) {
      console.error('Search error:', error);
      setError(error instanceof Error ? error.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  }

  const handleBack = () => {
    setAiResponse('')
  }

  return (
    <div className="flex flex-col min-h-screen">
      <Header {...headerConfig} currentLocale={`en`} currentType={`aiSearch`} />
      <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8">

        <div className="max-w-4xl mx-auto">
          <SearchHeader
            logo={searchConfig.logo}
            title={searchConfig.title}
            description={searchConfig.description}
          />
          <div className="flex flex-col sm:flex-row space-y-4 sm:space-y-0 sm:space-x-4 justify-center items-center mb-12">
            <SearchBar
              query={query}
              onQueryChange={setQuery}
              onSubmit={handleSearch}
            />
            <LegacyDocsToggle />
          </div>

          <AnimatePresence>
            {isLoading ? (
              <div className="flex flex-col justify-center items-center space-y-4 py-12">
                <AILoader />
                <p className="text-muted-foreground text-sm animate-pulse">
                  {loaderText}
                </p>
              </div>
            ) : error ? (
              <div className="text-center p-4 rounded-lg bg-red-50 text-red-800">
                <p className="text-lg font-medium mb-2">Error</p>
                <p>{error}</p>
              </div>
            ) : aiResponse ? (
              <AIResponse
                response={aiResponse}
                sources={sources}
                onBack={handleBack}
              />
            ) : recommendedArticles && (
              <RecommendedArticles articles={recommendedArticles} />
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  )
}
