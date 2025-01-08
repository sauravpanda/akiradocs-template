"use client";

import { useState, KeyboardEvent, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Folder,
  File,
  Check,
  X,
  ChevronRight,
  ChevronDown,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { fetchAllContent } from "@/lib/getContents";
import { FileExplorerContextMenu } from "@/components/context-menu";
import { LanguageSelector } from "./language-selector";
import { InsertionPoint } from "./insertion-point";
import * as React from "react";

type FileNode = {
  id: string;
  name: string;
  type: "file" | "folder";
  children?: FileNode[];
};

interface MetaItem {
  title: string
  path?: string
  items?: Record<string, MetaItem>
}

interface FileExplorerProps {
  onFileSelect: (path: string) => void;
}

interface RootMeta {
  defaultRoute?: string
  [key: string]: MetaItem | string | undefined
}

const API_URL =
  process.env.NEXT_PUBLIC_BACKEND_API_URL || "http://localhost:3001";

const getNodeFullPath = (
  tree: FileNode[],
  nodeId: string,
  parentPath: string = ""
): string | null => {
  for (const node of tree) {
    const currentPath = parentPath ? `${parentPath}/${node.name}` : node.name;
    if (node.id === nodeId) {
      return currentPath;
    }
    if (node.children) {
      const foundPath = getNodeFullPath(node.children, nodeId, currentPath);
      if (foundPath) return foundPath;
    }
  }
  return null;
};

export default function FileExplorer({ onFileSelect }: FileExplorerProps) {
  const [fileTree, setFileTree] = useState<FileNode[]>([]);
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(
    new Set(["1", "4"])
  );
  const [newItemParent, setNewItemParent] = useState<string | null>(null);
  const [newItemType, setNewItemType] = useState<"file" | "folder" | null>(
    null
  );
  const [newItemName, setNewItemName] = useState("");
  const [selectedLanguage, setSelectedLanguage] = useState("all");

  useEffect(() => {
    const content = fetchAllContent();
    const transformedTree = transformContentToFileTree(content);
    setFileTree(transformedTree);

    // Expand the selected language folder
    if (selectedLanguage !== "all") {
      const languageFolder = transformedTree.find(
        (node) => node.name === selectedLanguage
      );
      if (languageFolder) {
        setExpandedFolders((prev) => new Set(prev).add(languageFolder.id));
      }
    }
  }, [selectedLanguage]);

  useEffect(() => {
    if (selectedLanguage !== "all") {
      const languageFolder = fileTree.find(
        (node) => node.name === selectedLanguage
      );
      if (languageFolder) {
        setExpandedFolders((prev) => new Set(prev).add(languageFolder.id));
      }
    }
  }, [selectedLanguage, fileTree]);

  const toggleFolder = (folderId: string) => {
    setExpandedFolders((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(folderId)) {
        newSet.delete(folderId);
      } else {
        newSet.add(folderId);
      }
      return newSet;
    });
  };

  const handleFileClick = (node: FileNode) => {
    const fullPath = getNodeFullPath(fileTree, node.id);
    if (!fullPath) {
      console.error("Could not find full path for node");
      return;
    }
    onFileSelect(fullPath);
  };

  const startNewItem = (parentId: string, type: "file" | "folder") => {
    console.log("Starting new item:", { parentId, type }); // Add debug logging
    setNewItemParent(parentId);
    setNewItemType(type);
    setNewItemName("");
  };

  const cancelNewItem = () => {
    setNewItemParent(null);
    setNewItemType(null);
    setNewItemName("");
  };

  const addNewItem = async () => {
    if (!newItemParent || !newItemType || !newItemName) return;

    const newItem: FileNode = {
      id: Date.now().toString(),
      name: newItemName,
      type: newItemType,
      children: newItemType === "folder" ? [] : undefined,
    };

    const parentPath = getNodeFullPath(fileTree, newItemParent);
    if (!parentPath) {
      console.error("Could not find parent path");
      return;
    }

    const fullPath = `${parentPath}/${newItemName}`;

    try {
      if (newItemType === "file") {
         // Get the language and section from the path
         const pathParts = parentPath.split('/')
         const language = pathParts[0] // e.g. 'en'
         const section = pathParts[1] // e.g. 'docs'
         
         // Create default content for the new file
         const fileId = newItemName.replace('.json', '')
         const defaultContent = {
          id: fileId,
          title: fileId.split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' '),
          description: "",
          author: "Anonymous",
          date: new Date().toISOString().split("T")[0],
          blocks: [],
        };

        const fileResponse = await fetch(`${API_URL}/api/files`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            path: fullPath,
            content: defaultContent,
            isDirectory: false,
          }),
        });

        if (!fileResponse.ok) {
          const errorText = await fileResponse.text();
          console.error("File creation failed:", errorText);
          throw new Error(`Failed to create file: ${errorText}`);
        }

        await updateMetadata(language, section, pathParts, fileId, defaultContent);
      } else if (newItemType === "folder") {
        const folderResponse = await fetch(`${API_URL}/api/files`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            path: fullPath,
            isDirectory: true,
          }),
        });

        if (!folderResponse.ok) {
          const errorText = await folderResponse.text();
          console.error("Folder creation failed:", errorText);
          throw new Error(`Failed to create folder: ${errorText}`);
        }
      }

      const updatedTree = addItemToTree(fileTree, newItemParent, newItem);
      setFileTree(updatedTree);

      if (newItemType === "folder") {
        setExpandedFolders((prev) => new Set(prev).add(newItem.id));
      }

      cancelNewItem();
    } catch (error) {
      console.error("addNewItem: Error creating item:", error);
    }
  };

  const updateMetadata = async (
    language: string,
    section: string,
    pathParts: string[],
    fileId: string,
    defaultContent: any
  ) => {

    try {
      const rootMetaPath = `${language}/${section}/_meta.json`;
      const rootMetaResponse = await fetch(`${API_URL}/api/files?path=${encodeURIComponent(rootMetaPath)}`)

    if (rootMetaResponse.ok) {
      const rootMeta = await rootMetaResponse.json() as RootMeta
    
       // Navigate through the path to find the right section
       let currentSection = rootMeta
       for (let i = 2; i < pathParts.length; i++) {
         const part = pathParts[i]
         
         // Convert path to camelCase for section key
         const sectionKey = part.replace(/-/g, ' ')
           .split(' ')
           .map((word, index) => {
             const capitalized = word.charAt(0).toUpperCase() + word.slice(1)
             return index === 0 ? capitalized.toLowerCase() : capitalized
           })
           .join('')

         // Create section if it doesn't exist
         if (!currentSection[sectionKey]) {
           const newSection: MetaItem = {
             title: part.split('-')
               .map(word => word.charAt(0).toUpperCase() + word.slice(1))
               .join(' '),
             items: {}
           }
           currentSection[sectionKey] = newSection
         }
         
         // Move to items object for next iteration
         const section = currentSection[sectionKey] as MetaItem
         const items = section?.items
         if (!items) {
           currentSection[sectionKey] = {
             ...section,
             items: {}
           }
           currentSection = currentSection[sectionKey].items!
         } else {
           currentSection = items
         }
       }

       // Add the new file entry
       currentSection[fileId] = {
          title: defaultContent.title,
          path: `/${section}/${pathParts.slice(2).join('/')}${pathParts.slice(2).length > 0 ? '/' : ''}${fileId}`
        }

        // Save updated root meta
        await fetch(`${API_URL}/api/files`, {
          method: 'POST', 
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            path: rootMetaPath,
            content: rootMeta
          })
        })
      }
    } catch (error) {
      console.error("Error updating metadata:", error);
    }
  };

  const addItemToTree = (
    tree: FileNode[],
    parentId: string,
    newItem: FileNode
  ): FileNode[] => {
    return tree.map((node) => {
      if (node.id === parentId) {
        return {
          ...node,
          children: [...(node.children || []), newItem],
        };
      }
      if (node.children) {
        return {
          ...node,
          children: addItemToTree(node.children, parentId, newItem),
        };
      }
      return node;
    });
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      addNewItem();
    }
  };
  const filterTreeByLanguage = (nodes: FileNode[]): FileNode[] => {
    if (selectedLanguage === "all") return nodes;

    const languageFolder = nodes.find((node) => node.name === selectedLanguage);
    if (languageFolder && languageFolder.children) {
      return languageFolder.children;
    }

    return [];
  };

  const renderFileTree = (nodes: FileNode[], level: number = 0, parentId?: string) => {
    return (
      <ul className={`space-y-1 ${level > 0 ? "ml-4 pl-4" : ""}`}>
        {nodes.map((node, index) => (
          <React.Fragment key={node.id}>
            <li className="relative">
              <FileExplorerContextMenu
                onNewFile={() => startNewItem(node.id, "file")}
                onNewFolder={() => startNewItem(node.id, "folder")}
                onDelete={() => deleteItem(node.id, node.name, node.type)}
                isFolder={node.type === "folder"}
                disableDelete={
                  node.type === "folder" && (
                    level === 0 || // Language folders (en, es, etc)
                    (level === 1 && ["docs", "articles", "api"].includes(node.name)) // Protected subfolders
                  )
                }
              >
                <div className="flex items-center justify-between py-1">
                  <div className="flex items-center flex-grow">
                    {node.type === "folder" && (
                      <button
                        onClick={() => toggleFolder(node.id)}
                        className="mr-1 focus:outline-none"
                        aria-label={
                          expandedFolders.has(node.id)
                            ? "Collapse folder"
                            : "Expand folder"
                        }
                      >
                        {expandedFolders.has(node.id) ? (
                          <ChevronDown className="w-4 h-4 text-muted-foreground" />
                        ) : (
                          <ChevronRight className="w-4 h-4 text-muted-foreground" />
                        )}
                      </button>
                    )}
                    <div className="h-4 w-4 mr-1">
                      {node.type === "folder" ? (
                        <Folder className="h-4 w-4 text-muted-foreground" />
                      ) : (
                        <File className="h-4 w-4 text-muted-foreground" />
                      )}
                    </div>
                    <span
                      className={`text-sm ${
                        node.type === "folder" ? "font-semibold" : ""
                      } text-foreground hover:text-primary transition-colors duration-200 cursor-pointer`}
                      onClick={() =>
                        node.type === "file"
                          ? handleFileClick(node)
                          : toggleFolder(node.id)
                      }
                    >
                      {node.type === "file"
                        ? node.name.split(".").slice(0, -1).join(".")
                        : node.name}
                    </span>
                  </div>
                </div>
              </FileExplorerContextMenu>
              {node.type === "folder" && node.children && (
                <AnimatePresence>
                  {expandedFolders.has(node.id) && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                      transition={{ duration: 0.2 }}
                    >
                      <div className="mt-1">
                        {renderFileTree(node.children, level + 1, node.id)}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              )}
              {newItemParent === node.id && (
                <div className="flex items-center mt-2 pr-8">
                  <Input
                    type="text"
                    value={newItemName}
                    onChange={(e) => setNewItemName(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder={`New ${newItemType}`}
                    className="h-8 text-sm bg-background text-foreground flex-grow"
                  />
                  <Button
                    onClick={addNewItem}
                    size="sm"
                    className="ml-2 bg-secondary hover:bg-secondary/80 text-secondary-foreground"
                  >
                    <Check className="w-4 h-4" />
                  </Button>
                  <Button
                    onClick={cancelNewItem}
                    size="sm"
                    variant="ghost"
                    className="ml-1 text-muted-foreground"
                  >
                    <X className="w-4 h-4" />
                  </Button>
                </div>
              )}
            </li>
            <InsertionPoint
              onNewFile={() => startNewItem(parentId || node.id, "file")}
              onNewFolder={() => startNewItem(parentId || node.id, "folder")}
            />
          </React.Fragment>
        ))}
      </ul>
    );
  };
  const transformContentToFileTree = (content: {
    [key: string]: any;
  }): FileNode[] => {
    const tree: { [key: string]: FileNode } = {};
    let rootNodes: FileNode[] = [];

    Object.keys(content).forEach((path) => {
      const parts = path.split("/");
      let currentPath = "";

      parts.forEach((part, index) => {
        const isFile = index === parts.length - 1;
        const fullPath = currentPath ? `${currentPath}/${part}` : part;
        const nodeId = fullPath.replace(/[/.]/g, "_");

        if (!tree[fullPath]) {
          tree[fullPath] = {
            id: nodeId,
            name: part,
            type: isFile ? "file" : "folder",
            children: isFile ? undefined : [],
          };
        }

        if (index === 0) {
          if (!rootNodes.find((node) => node.id === nodeId)) {
            rootNodes.push(tree[fullPath]);
          }
        } else {
          const parentPath = currentPath;
          const parent = tree[parentPath];
          if (
            parent &&
            parent.children &&
            !parent.children.find((child) => child.id === nodeId)
          ) {
            parent.children.push(tree[fullPath]);
          }
        }

        currentPath = fullPath;
      });
    });

    return rootNodes;
  };

  const deleteItem = async (
      nodeId: string,
      nodeName: string,
      nodeType: 'file' | 'folder'
    ) => {
      const confirmMessage = `Are you sure you want to delete this ${nodeType}${
        nodeType === 'folder' ? ' and all its contents' : ''
      }?`;
      if (!confirm(confirmMessage)) {
        return;
      }

      const fullPath = getNodeFullPath(fileTree, nodeId);
      if (!fullPath) {
        console.error('Could not find full path for node');
        return;
      }

      try {
        console.log('Deleting item from tree:', { nodeName, fullPath });
        const response = await fetch(`${API_URL}/api/files`, {
          method: 'DELETE',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            path: fullPath,
            type: nodeType,
          }),
        });

        if (!response.ok) {
          throw new Error('Failed to delete item');
        }

        // Update metadata
        const pathParts = fullPath.split('/');
        const language = pathParts[0];
        const section = pathParts[1];
        const fileId = nodeName.replace('.json', '');

        // Get current metadata
        const rootMetaPath = `${language}/${section}/_meta.json`;
        const metaResponse = await fetch(
          `${API_URL}/api/files?path=${encodeURIComponent(rootMetaPath)}`
        );

        if (metaResponse.ok) {
          const rootMeta = (await metaResponse.json()) as RootMeta;

          // Navigate through the path to find the right section
          let currentSection = rootMeta;
          for (let i = 2; i < pathParts.length - 1; i++) {
            const part = pathParts[i];

            // Convert path to camelCase for section key
            const sectionKey = part
              .replace(/-/g, ' ')
              .split(' ')
              .map((word, index) => {
                const capitalized =
                  word.charAt(0).toUpperCase() + word.slice(1);
                return index === 0 ? capitalized.toLowerCase() : capitalized;
              })
              .join('');

            // For deletion, we only need to delete the key itself,
            // whether it's a file or folder
            if (i === pathParts.length - 2) {
              // We've reached the parent level where we need to delete
              if (currentSection[sectionKey]) {
                delete currentSection[sectionKey];
              }
              break;
            } else if (
              currentSection[sectionKey] &&
              (currentSection[sectionKey] as MetaItem).items
            ) {
              currentSection = (currentSection[sectionKey] as MetaItem).items!;
            }
          }

          // If it's a top-level item
          if (pathParts.length <= 3) {
            const key = fileId
              .replace(/-/g, ' ')
              .split(' ')
              .map((word, index) => {
                const capitalized =
                  word.charAt(0).toUpperCase() + word.slice(1);
                return index === 0 ? capitalized.toLowerCase() : capitalized;
              })
              .join('');

            if (currentSection[key]) {
              delete currentSection[key];
            }
          }

          // Save updated metadata
          await fetch(`${API_URL}/api/files`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              path: rootMetaPath,
              content: rootMeta,
            }),
          });
        }

        const updatedTree = deleteItemFromTree(fileTree, nodeId);
        setFileTree(updatedTree);
      } catch (error) {
        console.error('Error deleting item:', error);
      }
    };

  const deleteItemFromTree = (tree: FileNode[], nodeId: string): FileNode[] => {
    return tree.filter((node) => {
      if (node.id === nodeId) {
        return false;
      }
      if (node.children) {
        node.children = deleteItemFromTree(node.children, nodeId);
      }
      return true;
    });
  };

  return (
    <div className="px-8 py-8 bg-background min-h-screen text-foreground">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-3xl font-bold text-foreground">Project Explorer</h1>
        <LanguageSelector
          value={selectedLanguage}
          onValueChange={setSelectedLanguage}
        />
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        {renderFileTree(filterTreeByLanguage(fileTree))}
      </motion.div>
    </div>
  );
}

