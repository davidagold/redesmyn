import { Fragment, type ReactNode } from "react"

type MarkdownProps = {
  content: string
  omitFirstHeading?: boolean
}

type Block = {
  type: "heading"
  level: number
  text: string
} | {
  type: "paragraph"
  lines: string[]
} | {
  type: "list"
  ordered: boolean
  items: string[]
} | {
  type: "code"
  language: string | null
  code: string
} | { type: "hr" }

function parseMarkdown(content: string): Block[] {
  const lines = content.replaceAll("\r\n", "\n").split("\n")
  const blocks: Block[] = []
  let paragraph: string[] = []

  function flushParagraph() {
    if (!paragraph.length) {
      return
    }
    blocks.push({ type: "paragraph", lines: paragraph })
    paragraph = []
  }

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i] ?? ""

    const fenceMatch = line.match(/^```(.*)$/)
    if (fenceMatch) {
      flushParagraph()
      const language = fenceMatch[1]?.trim() || null
      const codeLines: string[] = []
      for (i += 1; i < lines.length; i += 1) {
        const codeLine = lines[i] ?? ""
        if (codeLine.startsWith("```")) {
          break
        }
        codeLines.push(codeLine)
      }
      blocks.push({ type: "code", language, code: codeLines.join("\n") })
      continue
    }

    const headingMatch = line.match(/^(#{1,6})\s+(.*)$/)
    if (headingMatch) {
      flushParagraph()
      blocks.push({
        type: "heading",
        level: headingMatch[1]?.length ?? 1,
        text: headingMatch[2] ?? "",
      })
      continue
    }

    if (line.match(/^\s*([-*_]){3,}\s*$/)) {
      flushParagraph()
      blocks.push({ type: "hr" })
      continue
    }

    const unorderedMatch = line.match(/^[-*]\s+(.*)$/)
    if (unorderedMatch) {
      flushParagraph()
      const items: string[] = []
      items.push(unorderedMatch[1] ?? "")
      for (i += 1; i < lines.length; i += 1) {
        const next = lines[i] ?? ""
        const match = next.match(/^[-*]\s+(.*)$/)
        if (!match) {
          i -= 1
          break
        }
        items.push(match[1] ?? "")
      }
      blocks.push({ type: "list", ordered: false, items })
      continue
    }

    const orderedMatch = line.match(/^\d+\.\s+(.*)$/)
    if (orderedMatch) {
      flushParagraph()
      const items: string[] = []
      items.push(orderedMatch[1] ?? "")
      for (i += 1; i < lines.length; i += 1) {
        const next = lines[i] ?? ""
        const match = next.match(/^\d+\.\s+(.*)$/)
        if (!match) {
          i -= 1
          break
        }
        items.push(match[1] ?? "")
      }
      blocks.push({ type: "list", ordered: true, items })
      continue
    }

    if (!line.trim()) {
      flushParagraph()
      continue
    }

    paragraph.push(line)
  }

  flushParagraph()
  return blocks
}

function renderInline(text: string): ReactNode[] {
  const pattern = /(`[^`]+`)|(\[[^\]]+\]\([^)]+\))|(\*\*[^*]+\*\*)|(\*[^*]+\*)/g
  const parts: ReactNode[] = []
  let lastIndex = 0

  for (const match of text.matchAll(pattern)) {
    const start = match.index ?? 0
    if (start > lastIndex) {
      parts.push(text.slice(lastIndex, start))
    }

    const token = match[0] ?? ""
    const key = `${start}:${token}`

    if (token.startsWith("`") && token.endsWith("`")) {
      parts.push(
        <code
          key={key}
          className="rounded-sm bg-muted px-1 py-0.5 font-mono text-[0.8em] text-foreground"
        >
          {token.slice(1, -1)}
        </code>,
      )
    } else if (token.startsWith("[")) {
      const linkMatch = token.match(/^\[([^\]]+)\]\(([^)]+)\)$/)
      const label = linkMatch?.[1] ?? token
      const href = linkMatch?.[2] ?? "#"
      parts.push(
        <a
          key={key}
          href={href}
          target="_blank"
          rel="noreferrer"
          className="text-primary underline underline-offset-4 hover:opacity-90"
        >
          {label}
        </a>,
      )
    } else if (token.startsWith("**") && token.endsWith("**")) {
      parts.push(
        <strong key={key} className="font-semibold text-foreground">
          {token.slice(2, -2)}
        </strong>,
      )
    } else if (token.startsWith("*") && token.endsWith("*")) {
      parts.push(
        <em key={key} className="italic text-foreground">
          {token.slice(1, -1)}
        </em>,
      )
    } else {
      parts.push(token)
    }

    lastIndex = start + token.length
  }

  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex))
  }

  return parts
}

export function Markdown({ content, omitFirstHeading }: MarkdownProps) {
  let blocks = parseMarkdown(content)
  if (
    omitFirstHeading &&
    blocks[0]?.type === "heading" &&
    blocks[0].level === 1
  ) {
    blocks = blocks.slice(1)
  }

  return (
    <div className="grid gap-3 text-sm leading-relaxed text-foreground">
      {blocks.map((block, idx) => {
        const key = `${block.type}:${idx}`
        switch (block.type) {
          case "heading": {
            const level = Math.min(Math.max(block.level, 1), 6)
            const text = renderInline(block.text)
            const className =
              level === 1
                ? "text-lg font-semibold tracking-tight"
                : level === 2
                  ? "text-base font-semibold"
                  : "text-sm font-semibold"

            const H =
              level === 1
                ? "h1"
                : level === 2
                  ? "h2"
                  : level === 3
                    ? "h3"
                    : level === 4
                      ? "h4"
                      : level === 5
                        ? "h5"
                        : "h6"

            return (
              <H key={key} className={className}>
                {text}
              </H>
            )
          }
          case "paragraph":
            return (
              <p key={key} className="text-foreground/90">
                {renderInline(block.lines.join(" "))}
              </p>
            )
          case "hr":
            return <hr key={key} className="border-border" />
          case "list": {
            const List = block.ordered ? "ol" : "ul"
            return (
              <List
                key={key}
                className={`grid gap-1 pl-5 text-foreground/90 ${
                  block.ordered ? "list-decimal" : "list-disc"
                }`}
              >
                {block.items.map((item, itemIdx) => (
                  <li key={`${key}:${itemIdx}`}>{renderInline(item)}</li>
                ))}
              </List>
            )
          }
          case "code":
            return (
              <div key={key} className="rounded-md border bg-muted p-3">
                {block.language ? (
                  <div className="mb-2 font-mono text-xs text-muted-foreground">
                    {block.language}
                  </div>
                ) : null}
                <pre className="overflow-auto whitespace-pre-wrap font-mono text-xs leading-relaxed text-foreground">
                  <code>{block.code}</code>
                </pre>
              </div>
            )
          default:
            return (
              <Fragment key={key}>{JSON.stringify(block) as string}</Fragment>
            )
        }
      })}
    </div>
  )
}
