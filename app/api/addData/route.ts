import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { NextRequest, NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import { OpenAIEmbeddings } from "@langchain/openai";
import { PineconeStore } from "@langchain/community/vectorstores/pinecone";

export async function POST(request: NextRequest) {
  // Extract FormData from the request
  const data = await request.formData();
  // Extract the uploaded file from the FormData
  const file: File | null = data.get("file") as unknown as File;

  // Make sure file exists
  if (!file) {
    return NextResponse.json({ success: false, error: "No file found" });
  }

  // Make sure file is a PDF
  if (file.type !== "application/pdf") {
    return NextResponse.json({ success: false, error: "Invalid file type" });
  }

  // Use the PDFLoader to load the PDF and split it into smaller documents
  const pdfLoader = new PDFLoader(file);
  const splitDocuments = await pdfLoader.loadAndSplit();

  // Initialize the Pinecone client
  const pinecone = new Pinecone({
    apiKey:process.env.PINECONE_API_KEY as string,
    environment: "gcp-starter",
  });

  
  const pineconeIndex = pinecone.Index(
    process.env.PINECONE_INDEX_NAME as string
  );

  // Use Langchain's integration with Pinecone to store the documents
  await PineconeStore.fromDocuments(splitDocuments, new OpenAIEmbeddings(), {
    pineconeIndex,
  });

  return NextResponse.json({ success: true });
}