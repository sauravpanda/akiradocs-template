'use client'

import React, { useState, useCallback } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronRight, FileText } from 'lucide-react'
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { NavigationProps, NavItemProps } from "@/types/navigation"
import { ErrorBoundary } from 'react-error-boundary'
import { getApiNavigation } from '@/lib/content';
import { useAnalytics } from '@/hooks/useAnalytics';
const buttonStyles = {
  base: "w-full justify-start text-left font-normal rounded-lg transition-all px-1.5 py-1 whitespace-normal break-words",
  hover: "hover:bg-accent/40 hover:text-accent-foreground",
  active: "w-full bg-gradient-to-r from-accent to-accent/80 text-accent-foreground font-medium shadow-md translate-x-1 px-1.5 py-1 break-words",
  state: "data-[state=open]:bg-accent/30",
}

function ErrorFallback({ error }: { error: Error }) {
  return (
    <div className="w-64 p-4 text-sm text-red-500">
      <h2 className="font-semibold">Navigation Error</h2>
      <p>Something went wrong loading the navigation.</p>
    </div>
  )
}

export function Navigation({ locale, items }: NavigationProps) {
  const pathname = usePathname()
  
  return (
    <ErrorBoundary FallbackComponent={ErrorFallback}>
      <aside className="w-[20%] bg-sidebar-background/50 text-sidebar-foreground border-r border-border/40 h-[calc(100vh-4rem)] sticky top-16 backdrop-blur-sm overflow-hidden">
        <ScrollArea className="h-full py-6 px-4">
          <nav className="space-y-2">
            {Object.entries(items)
              .filter(([key]) => key !== "defaultRoute")
              .map(([key, item]) => (
                <NavItem key={key} locale={locale} item={item} pathname={pathname} />
              ))}
          </nav>
        </ScrollArea>
      </aside>
    </ErrorBoundary>
  )
}

const NavItem = React.memo(({ locale, item, pathname, depth = 0 }: NavItemProps) => {
  const [isOpen, setIsOpen] = useState(true)
  const hasChildren = item.items && Object.keys(item.items).length > 0
  const isActive = item.path ? pathname === `/${locale}${item.path}` : false
  const absolutePath = item.path ? `/${locale}${item.path}` : '#'
  const { track } = useAnalytics()
  const itemRef = React.useRef<HTMLDivElement>(null)

  React.useEffect(() => {
    if (isActive && itemRef.current) {
      setTimeout(() => {
        const viewport = itemRef.current?.closest('[data-radix-scroll-area-viewport]') as HTMLElement;
        if (viewport) {
          const itemRect = itemRef.current?.getBoundingClientRect();
          const viewportRect = viewport.getBoundingClientRect();
          const scrollOffset = (itemRect?.top ?? 0) - (viewportRect.top ?? 0) - (viewportRect.height / 2) + ((itemRect?.height ?? 0) / 2);
          
          viewport.scrollTop += scrollOffset;
        }
      }, 100);
    }
  }, [isActive]);

  const handleClick = useCallback((e: React.MouseEvent) => {
    track('navigation_click', {
      page_path: pathname,
      navigation_item: item.title
    })
    if (item.path && pathname === `/${locale}${item.path}`) {
      e.preventDefault()
    }
    if (hasChildren) {
      setIsOpen(prev => !prev)
    }
  }, [hasChildren, item.path, locale, pathname])

  return (
    <motion.div 
      ref={itemRef}
      className={cn(
        "relative mb-1",
        depth > 0 && "ml-3 border-l border-border/50 pl-3 before:absolute before:left-0 before:top-0 before:bottom-0 before:w-px before:bg-gradient-to-b before:from-border/0 before:via-border/50 before:to-border/0"
      )}
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.3 }}
    >
      <Button
        variant="ghost"
        className={cn(
          buttonStyles.base,
          buttonStyles.hover,
          buttonStyles.state,
          isActive && buttonStyles.active,
          depth > 0 && "text-sm text-muted-foreground",
          "group flex items-start gap-2",
          "min-h-[2.5rem] h-auto"
        )}
        onClick={handleClick}
      >
        {hasChildren && (
          <motion.div
            className="text-muted-foreground/70 group-hover:text-muted-foreground shrink-0 mt-1"
            initial={false}
            animate={{ rotate: isOpen ? 90 : 0 }}
            transition={{ duration: 0.2 }}
          >
            <ChevronRight className="h-3.5 w-3.5" />
          </motion.div>
        )}
        {item.path ? (
          <Link 
            href={absolutePath} 
            className="flex-1 leading-tight py-0.5"
            onClick={handleClick}
          >
            <span className="inline-block">{item.title}</span>
          </Link>
        ) : (
          <span className="flex-1 leading-tight py-0.5 inline-block">{item.title}</span>
        )}
        {item.badge && (
          <span className="px-2 py-0.5 text-xs rounded-full bg-accent text-accent-foreground shrink-0 self-start mt-1">
            {item.badge}
          </span>
        )}
      </Button>
      <AnimatePresence initial={false}>
        {hasChildren && isOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3, ease: "easeInOut" }}
            className="mt-1"
          >
            {Object.entries(item.items!).map(([key, child]) => (
              <NavItem locale={locale} key={key} item={child} pathname={pathname} depth={depth + 1} />
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
})

NavItem.displayName = 'NavItem'

export async function ApiSidebar() {
  const navigation = await getApiNavigation();
  
  return (
    <ErrorBoundary FallbackComponent={ErrorFallback}>
      <aside className="w-[20%] bg-sidebar-background text-sidebar-foreground border-r h-[calc(100vh-4rem)] sticky top-16 shadow-sm">
        <ScrollArea className="h-full py-6 px-4">
          <nav className="space-y-2">
            {navigation.map((item, index) => (
              <ApiNavItem key={index} item={item} />
            ))}
          </nav>
        </ScrollArea>
      </aside>
    </ErrorBoundary>
  );
}

const ApiNavItem = React.memo(({ item }: { item: any }) => {
  const methodClass = item.method.toLowerCase()
  const sectionId = `${methodClass}-${item.path}`

  const scrollToSection = useCallback((elementId: string) => {
    const element = document.getElementById(elementId)
    if (element) {
      const headerOffset = 64 + 32
      const elementPosition = element.getBoundingClientRect().top
      const offsetPosition = elementPosition + window.pageYOffset - headerOffset

      window.scrollTo({
        top: offsetPosition,
        behavior: 'smooth'
      })
    }
  }, [])

  // Define badge colors to match main content
  const methodColors = {
    get: 'bg-green-100 text-green-800',
    post: 'bg-blue-100 text-blue-800',
    put: 'bg-yellow-100 text-yellow-800',
    patch: 'bg-orange-100 text-orange-800',
    delete: 'bg-red-100 text-red-800',
  }

  return (
    <motion.div 
      className="mb-1"
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.3 }}
    >
      <Button
        variant="ghost"
        className={cn(
          buttonStyles.base,
          buttonStyles.hover,
          buttonStyles.state,
        )}
        onClick={() => scrollToSection(sectionId)}
      >
        <div className="flex-1">
          <span className={`mr-2 px-2 py-1 text-xs font-medium rounded-md ${methodColors[methodClass as keyof typeof methodColors] || 'bg-gray-100 text-gray-800'}`}>
            {item.method.toUpperCase()}
          </span>
          {item.title}
        </div>
      </Button>
    </motion.div>
  )
})

ApiNavItem.displayName = 'ApiNavItem'

export default Navigation
