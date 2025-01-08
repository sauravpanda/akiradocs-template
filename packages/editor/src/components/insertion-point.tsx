"use client";

import { useState, useRef, useEffect } from "react";
import { Plus, File, Folder } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

export function InsertionPoint({
  onNewFile,
  onNewFolder,
}: {
  onNewFile: () => void;
  onNewFolder: () => void;
}) {
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setMenuOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  const toggleMenu = () => {
    setMenuOpen((prev) => !prev);
  };

  return (
    <div className="group relative py-1 -my-1">
      <div className="absolute inset-x-0 h-0.5 scale-x-0 bg-border group-hover:scale-x-100 transition-transform" />
      <Button
        variant="ghost"
        size="icon"
        className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 h-5 w-5 rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
        aria-label="Add item"
        onClick={toggleMenu}
      >
        <Plus className="h-3 w-3" />
      </Button>

      {menuOpen && (
        <Card
          ref={menuRef}
          className="absolute left-1/2 top-full transform -translate-x-1/2 mt-2 w-48 z-50 bg-popover border-border p-0.5"
        >
          <Button
            variant="ghost"
            className="w-full justify-start h-8 px-2 text-sm font-normal"
            onClick={() => {
              onNewFile();
              setMenuOpen(false);
            }}
          >
            <File className="mr-2 h-3.5 w-3.5" />
            New File
          </Button>
          <Button
            variant="ghost"
            className="w-full justify-start h-8 px-2 text-sm font-normal"
            onClick={() => {
              onNewFolder();
              setMenuOpen(false);
            }}
          >
            <Folder className="mr-2 h-3.5 w-3.5" />
            New Folder
          </Button>
        </Card>
      )}
    </div>
  );
}
