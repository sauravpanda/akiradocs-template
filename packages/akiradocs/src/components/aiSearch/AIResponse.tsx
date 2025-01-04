"use client"
import { Sparkles, ArrowLeft } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { motion } from "framer-motion"
import { AIResponseActions } from "./AIResponseActions"
import { AIResponseSources } from "./AIResponseSources"
import ReactMarkdown from 'react-markdown'
import { Source } from "@/types/Source"
import { memo } from 'react'

interface AIResponseProps {
  response: string
  sources: Source[]
  onBack: () => void
}

// Memoize the markdown component to prevent unnecessary re-renders
const MemoizedMarkdown = memo(({ content }: { content: string }) => (
  <ReactMarkdown
    components={{
      pre({ children, ...props }) {
        return (
          <pre className="relative" {...props}>
            {children}
          </pre>
        )
      },
      p: ({ children }) => <p className="mb-4">{children}</p>,
      ol: ({ node, ...props }) => (
        <ol
          style={{
            display: "block",
            listStyleType: "decimal",
            paddingInlineStart: "40px",
          }}
          {...props}
        />
      ),
      ul: ({ node, ...props }) => (
        <ul
          style={{
            display: "block",
            listStyleType: "disc",
            paddingInlineStart: "40px",
          }}
          {...props}
        />
      ),
      br: ({ node, ...props }) => (
        <br className="mb-2" {...props} />
      ),
      code({ node, inline, className, children, ...props }) {
        const match = /language-(\w+)/.exec(className || '')
        return inline ? (
          <code className={className} {...props}>
            {children}
          </code>
        ) : (
          <div className="relative">
            <code className={`language-${match?.[1] || 'text'}`} {...props}>
              {children}
            </code>
          </div>
        )
      }
    }}
  >
    {content}
  </ReactMarkdown>
))
MemoizedMarkdown.displayName = 'MemoizedMarkdown'

export function AIResponse({ response, sources, onBack }: AIResponseProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.5 }}
    >
      <Card className="mt-8 shadow-lg border-border">
        <CardHeader className="bg-accent/50 flex flex-row items-center justify-between">
          <CardTitle className="flex items-center text-2xl text-primary">
            <Sparkles className="mr-2 h-6 w-6" />
            AI Guide
          </CardTitle>
          <Button variant="ghost" size="sm" onClick={onBack} className="text-primary hover:text-primary/80">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back
          </Button>
        </CardHeader>
        <CardContent className="pt-2">
          <div className="prose dark:prose-invert max-w-none
            prose-headings:font-semibold
            prose-h1:text-xl prose-h1:mt-8 prose-h1:mb-4
            prose-h2:text-lg prose-h2:mt-6 prose-h2:mb-3
            prose-p:my-4 prose-p:leading-relaxed
            prose-ul:space-y-2 prose-ul:list-disc prose-ul:pl-6
            prose-ol:space-y-2 prose-ol:list-decimal prose-ol:pl-6
            prose-li:pl-2
            [&_pre]:bg-secondary [&_pre]:p-4 [&_pre]:rounded-lg [&_pre]:my-4
            [&_code:not(pre_code)]:bg-secondary [&_code:not(pre_code)]:px-1.5 [&_code:not(pre_code)]:py-0.5 [&_code:not(pre_code)]:rounded-sm
            [&_pre_code]:bg-transparent [&_pre_code]:p-0 [&_pre_code]:rounded-none
            [&>*:first-child]:mt-0
            [&>p>strong]:block [&>p>strong]:mt-8 [&>p>strong]:mb-4 [&>p>strong]:text-lg
            [&>p:has(>strong:only-child)]:m-0">
            <MemoizedMarkdown content={response} />
          </div>
        </CardContent>
        <CardFooter className="flex flex-col items-start space-y-4">
          <AIResponseActions response={response} />
          <Separator className="my-4" />
          <AIResponseSources sources={sources} />
        </CardFooter>
      </Card>
    </motion.div>
  )
}