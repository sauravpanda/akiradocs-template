import { writeFile, mkdir, rm, unlink, readFile } from "fs/promises";
import { NextResponse } from "next/server";
import path from "path";

function getBasePath() {
  return path.join(process.cwd(), "..", "compiled");
}

export async function POST(request: Request) {
  try {
    const { path: filePath, content, isDirectory } = await request.json();
    const fullPath = path.join(getBasePath(), filePath);

    // Ensure the directory exists
    await mkdir(path.dirname(fullPath), { recursive: true });

    if (isDirectory) {
      // If it's a directory, just create it
      await mkdir(fullPath, { recursive: true });
    } else {
      // If it's a file, write the content
      await writeFile(fullPath, JSON.stringify(content, null, 2));
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Error creating file:", error);
    return NextResponse.json(
      { error: "Failed to create file" },
      { status: 500 }
    );
  }
}

export async function DELETE(request: Request) {
  try {
    const { path: itemPath, type } = await request.json();
    const fullPath = path.join(getBasePath(), itemPath);

    if (type === "folder") {
      await rm(fullPath, { recursive: true, force: true });
    } else {
      await unlink(fullPath);
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Error deleting item:", error);
    return NextResponse.json(
      { error: "Failed to delete item" },
      { status: 500 }
    );
  }
}

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const filePath = searchParams.get("path");

    if (!filePath) {
      return NextResponse.json(
        { error: "No file path provided" },
        { status: 400 }
      );
    }

    const fullPath = path.join(getBasePath(), filePath);
    const fileContent = await readFile(fullPath, "utf-8");
    const parsedContent = JSON.parse(fileContent);

    return NextResponse.json(parsedContent);
  } catch (error) {
    console.error("Error reading file:", error);
    return NextResponse.json({ error: "Failed to read file" }, { status: 500 });
  }
}

export async function PUT(request: Request) {
  try {
    const { path: filePath, content } = await request.json();

    if (!filePath) {
      return NextResponse.json(
        { error: "No file path provided" },
        { status: 400 }
      );
    }

    // Validate content
    if (!content) {
      return NextResponse.json(
        { error: "No content provided" },
        { status: 400 }
      );
    }

    const fullPath = path.join(getBasePath(), filePath);

    // Check if directory exists, if not create it
    await mkdir(path.dirname(fullPath), { recursive: true });

    // Add retry logic
    const maxRetries = 3;
    for (let i = 0; i < maxRetries; i++) {
      try {
        const jsonContent = JSON.stringify(content, null, 2);
        await writeFile(fullPath, jsonContent, "utf-8");
        return NextResponse.json({ success: true });
      } catch (writeError) {
        if (i === maxRetries - 1) throw writeError;
        await new Promise((resolve) => setTimeout(resolve, 1000)); // Wait 1s before retry
      }
    }
  } catch (error) {
    console.error("Error writing file:", error);
    return NextResponse.json(
      {
        error: "Failed to write file",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}
