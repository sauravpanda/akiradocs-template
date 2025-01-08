import * as React from 'react';
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuTrigger,
} from '@/components/ui/context-menu';
import { File, Folder, Trash2 } from 'lucide-react';

interface FileExplorerContextMenuProps {
  children: React.ReactNode;
  onNewFile: () => void;
  onNewFolder: () => void;
  onDelete: () => void;
  isFolder: boolean;
  disableDelete?: boolean;
}

export function FileExplorerContextMenu({
  children,
  onNewFile,
  onNewFolder,
  onDelete,
  isFolder,
  disableDelete = false,
}: FileExplorerContextMenuProps) {
  return (
    <ContextMenu>
      <ContextMenuTrigger>{children}</ContextMenuTrigger>
      <ContextMenuContent className="w-64">
        {isFolder && (
          <>
            <ContextMenuItem onSelect={onNewFile}>
              <File className="mr-2 h-4 w-4" />
              New File
            </ContextMenuItem>
            <ContextMenuItem onSelect={onNewFolder}>
              <Folder className="mr-2 h-4 w-4" />
              New Folder
            </ContextMenuItem>
          </>
        )}
        {!disableDelete && (
          <ContextMenuItem onSelect={onDelete} className="text-red-600">
            <Trash2 className="mr-2 h-4 w-4" />
            Delete
          </ContextMenuItem>
        )}
      </ContextMenuContent>
    </ContextMenu>
  );
}

