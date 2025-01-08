'use client';

import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';
import { BlockType } from '@/types/Block';
import { Plus } from 'lucide-react';
import { ArticleHeaders } from '@/components/blocks/ArticleHeaders';
import { TitleBar } from '@/components/layout/TitleBar';
import {
  DndContext,
  DragEndEvent,
  MouseSensor,
  TouchSensor,
  useSensor,
  useSensors,
} from '@dnd-kit/core';
import {
  SortableContext,
  arrayMove,
  verticalListSortingStrategy,
} from '@dnd-kit/sortable';
import { SortableBlock } from '@/components/blocks/SortableBlock';
import { SEO } from '@/components/layout/SEO';

type Block = {
  id: string;
  type: BlockType;
  content: string;
  metadata?: Record<string, any>;
};

interface EditorProps {
  filePath: string;
}

export function Editor({ filePath }: EditorProps) {
  const [blocks, setBlocks] = useState<Block[]>([]);
  const [title, setTitle] = useState('');
  const [subtitle, setSubtitle] = useState('');
  const [showPreview, setShowPreview] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [activeChangeTypeId, setActiveChangeTypeId] = useState<string | null>(
    null
  );

  useEffect(() => {
    const loadFileContent = async () => {
      if (!filePath) {
        setBlocks([{ id: '1', type: 'paragraph', content: '', metadata: {} }]);
        setIsLoading(false);
        return;
      }

      try {
        const response = await fetch(
          `/api/files?path=${encodeURIComponent(filePath)}`
        );
        if (!response.ok) throw new Error('Failed to load file');
        const data = await response.json();
        setTitle(data.title || '');
        setSubtitle(data.description || '');
        setBlocks(
          data.blocks || [
            { id: '1', type: 'paragraph', content: '', metadata: {} },
          ]
        );
      } catch (error) {
        console.error('Error loading file:', error);
        setBlocks([{ id: '1', type: 'paragraph', content: '', metadata: {} }]);
      } finally {
        setIsLoading(false);
      }
    };

    loadFileContent();
  }, [filePath]);

  const handleSave = async () => {
    if (!filePath) {
      console.error('No file path specified');
      return;
    }

    setIsSaving(true);
    try {
      const content = {
        title,
        description: subtitle,
        author: 'Anonymous', // You might want to make this dynamic
        date: new Date().toISOString().split('T')[0],
        blocks,
      };

      const response = await fetch('/api/files', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: filePath, content }),
      });

      if (!response.ok) throw new Error('Failed to save file');
      toast.success('Changes saved successfully');
    } catch (error) {
      console.error('Error saving file:', error);
      toast.error('Failed to save changes');
    } finally {
      setIsSaving(false);
    }
  };

  const addBlock = (afterId: string) => {
    const newBlock: Block = {
      id: Date.now().toString(),
      type: 'paragraph',
      content: '',
      metadata: {},
    };

    if (newBlock.type === 'list') {
      newBlock.content = '[]';
    }

    if (afterId === 'new') {
      setBlocks([newBlock]);
    } else {
      const index = blocks.findIndex((block) => block.id === afterId);
      setBlocks([
        ...blocks.slice(0, index + 1),
        newBlock,
        ...blocks.slice(index + 1),
      ]);
    }
    setActiveChangeTypeId(newBlock.id);
  };

  const updateBlock = (id: string, content: string) => {
    setBlocks(
      blocks.map((block) => {
        if (block.id === id) {
          if (block.type === 'list') {
            try {
              const parsed = JSON.parse(content);
              return {
                ...block,
                content: JSON.stringify(
                  Array.isArray(parsed) ? parsed : [parsed]
                ),
              };
            } catch {
              return { ...block, content: JSON.stringify([content]) };
            }
          }
          return { ...block, content };
        }
        return block;
      })
    );
  };

  const changeBlockType = (id: string, newType: BlockType) => {
    setBlocks(
      blocks.map((block) =>
        block.id === id ? { ...block, type: newType } : block
      )
    );
    setActiveChangeTypeId(null);
  };

  const deleteBlock = (id: string) => {
    setBlocks(blocks.filter((block) => block.id !== id));
  };

  const sensors = useSensors(
    useSensor(MouseSensor, {
      activationConstraint: {
        distance: 8,
      },
    }),
    useSensor(TouchSensor, {
      activationConstraint: {
        delay: 200,
        tolerance: 8,
      },
    })
  );

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;

    if (over && active.id !== over.id) {
      setBlocks((blocks) => {
        const oldIndex = blocks.findIndex((block) => block.id === active.id);
        const newIndex = blocks.findIndex((block) => block.id === over.id);

        return arrayMove(blocks, oldIndex, newIndex);
      });
    }
  };

  const updateBlockMetadata = (id: string, metadata: any) => {
    setBlocks(
      blocks.map((block) => {
        if (block.id === id) {
          return { ...block, metadata: metadata };
        }
        return block;
      })
    );
  };

  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-screen">
        <div className="text-gray-500">Loading...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-4xl mx-auto px-4 py-8">
        <SEO
          title={`${title} | Editor`}
          description={subtitle}
          noIndex={true}
        />
        <TitleBar onSave={handleSave} isSaving={isSaving} />
        <div className="prose prose-lg max-w-none">
          <ArticleHeaders
            title={title}
            setTitle={setTitle}
            subtitle={subtitle}
            setSubtitle={setSubtitle}
            showPreview={showPreview}
          />
          <DndContext sensors={sensors} onDragEnd={handleDragEnd}>
            <SortableContext
              items={blocks}
              strategy={verticalListSortingStrategy}
            >
              {blocks.map((block) => (
                <SortableBlock
                  key={block.id}
                  block={block}
                  updateBlock={updateBlock}
                  changeBlockType={changeBlockType}
                  addBlock={addBlock}
                  deleteBlock={deleteBlock}
                  showPreview={showPreview}
                  isChangeTypeActive={activeChangeTypeId === block.id}
                  setActiveChangeTypeId={setActiveChangeTypeId}
                  updateBlockMetadata={updateBlockMetadata}
                />
              ))}
            </SortableContext>
          </DndContext>
          {blocks.length === 0 && !showPreview && (
            <div className="flex justify-center my-8">
              <Button
                onClick={() => addBlock('new')}
                variant="outline"
                className="flex items-center gap-2"
              >
                <Plus size={16} />
                Add Block
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default Editor;
