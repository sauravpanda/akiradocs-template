"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable";
const FileExplorer = dynamic(() => import("@/components/file-explorer"), {
  ssr: false,
});
const Editor = dynamic(() => import("@/components/editor"), { ssr: false });

export default function EditModePage() {
  const [selectedFile, setSelectedFile] = useState<string | null>(null);

  return (
    <ResizablePanelGroup direction="horizontal">
      <ResizablePanel defaultSize={20} minSize={15} maxSize={40}>
        <div className="h-full custom-scrollbar overflow-auto">
          <FileExplorer onFileSelect={setSelectedFile} />
        </div>
      </ResizablePanel>
      <ResizableHandle />
      <ResizablePanel defaultSize={80}>
        <div className="h-full custom-scrollbar overflow-auto">
          {selectedFile ? (
            <Editor filePath={selectedFile} />
          ) : (
            <div className="flex h-full items-center justify-center text-muted-foreground">
              Select a file to edit
            </div>
          )}
        </div>
      </ResizablePanel>
    </ResizablePanelGroup>
  );
}
