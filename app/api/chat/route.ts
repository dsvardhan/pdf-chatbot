import { NextRequest } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import { PineconeStore } from "@langchain/community/vectorstores/pinecone";
import { OpenAIEmbeddings } from "@langchain/openai";
import { OpenAI } from "@langchain/openai";
import { VectorDBQAChain } from "langchain/chains";
import { StreamingTextResponse, LangChainStream } from "ai";
import { CallbackManager } from  "@langchain/core/callbacks/manager";

export async function POST(request: NextRequest) {
  // Parse the POST request's JSON body
  const body = await request.json();
  
  // Use Vercel's `ai` package to setup a stream
  const { stream, handlers } = LangChainStream();

  // Initialize Pinecone Client
  //const pinecone = new Pinecone();
  const pinecone = new Pinecone({
    apiKey:process.env.PINECONE_API_KEY as string,
    environment: "gcp-starter",
  });

//   await pinecone.init({
//     environment: "gcp-starter",
//     apiKey: process.env.PINECONE_API_KEY,
    
//   });

  const pineconeIndex = pinecone.Index(
    "pdf-chatbot"
  );

  // Initialize our vector store
  const vectorStore = await PineconeStore.fromExistingIndex(
    new OpenAIEmbeddings(),
    { pineconeIndex }
  );

  // Specify the OpenAI model we'd like to use, and turn on streaming
  const model = new OpenAI({
    modelName: "gpt-3.5-turbo",
    streaming: true,
    callbackManager: CallbackManager.fromHandlers(handlers),
  });

  // Define the Langchain chain
  const chain = VectorDBQAChain.fromLLM(model, vectorStore, {
    k: 1,
    returnSourceDocuments: true,
  });

  // Call our chain with the prompt given by the user
  chain.call({ query: body.prompt }).catch(console.error);

  // Return an output stream to the frontend
  return new StreamingTextResponse(stream);
}