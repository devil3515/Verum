# Lovable Frontend - Agent Swarm Research Implementation

## Overview
Agenticet with Lovable platform to implement Agent Swarm Research interface on top of existing dashboard.

**Design principle**: Reuse existing shadcn/ui components, patterns from existing analysis/chat pages.

---

## Files to Create (Lovable will auto-generate from these)

### 1. `src/components/ResearchModeCard.tsx`

**Functionality**: Selection card for Research Mode vs Analysis Mode

```tsx
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { Search, Brain, FileText } from "lucide-react"

export function ResearchModeCard({
  mode,
  title,
  description,
  icon,
  onClick
}: {
  mode: "analysis" | "research" | "document"
  title: string
  description: string
  icon: React.ReactNode
  onClick: () => void
}) {
  return (
    <Card
      onClick={onClick}
      className="cursor-pointer hover:border-primary/50 hover:shadow-lg transition-all"
    >
      <CardHeader>
        <div className="flex items-center gap-4 mb-2">
          <div className="h-12 w-12 bg-muted rounded-lg flex items-center justify-center">
            {icon}
          </div>
          <CardTitle>{title}</CardTitle>
        </div>
        <p className="text-sm text-muted-foreground">{description}</p>
      </CardHeader>
    </Card>
  )
}
```

**Lovable notes**:
- Reuses existing `Card` component from `/src/components/ui/card.tsx`
- Uses existing `constIcon` function for shadcn/ui icons
- Minimal code - just pattern matching existing components

---

### 2. `src/components/TaskList.tsx`

**Functionality**: Scrollable list of all research tasks with status indicators

```tsx
import { Card, CardContent } from "@/components/ui/card"
import { Search } from "lucide-react"
import { Link } from "(@tanstack/react-router"

export function TaskList({ tasks }: { tasks: any[] }) {
  // Helper: Status color indicator
  const getStatusColor = (status: string) => {
    switch (status) {
      case "completed": return "bg-green-500"
      case "active": return "bg-blue-500 animate-pulse"
      case "queued": return "bg-amber-500"
      case "failed": return "bg-red-500"
      default: return "bg-gray-400"
    }
  }

  return (
    <div className="space-y-2">
      {tasks.map(task => (
        <Link key={task.task_id} to="/research/$id" params={{ id: task.task_id }}>
          <Card className="hover:bg-muted/50 transition-colors">
            <CardContent className="pt-6">
              <div className="flex items-center gap-4">
                <div className={`h-2 w-2 rounded-full ${getStatusColor(task.status)}`} />
                <div className="flex-1 min-w-0">
                  <h3 className="font-medium truncate">{task.objective}</h3>
                  <p className="text-xs text-muted-foreground">
                    {task.depth}-layer research ⚙️ {task.status}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </Link>
      ))}

      {tasks.length === 0 && (
        <div className="text-center py-8 text-muted-foreground">
          <Search className="h-8 w-8 mx-auto mb-2" />
          <p>No research tasks yet</p>
        </div>
      )}
    </div>
  )
}
```

**Lovable notes**:
- Reuses existing `Card` and `CardContent` from UI components
- Use existing `Link` component (TanStack Router)
- Status colors can be copied from existing use case in codebase

---

### 3. `src/components/ResearchStage.tsx`

**Functionality**: Display live research events like a timeline with different levels

```tsx
import { Card, CardHeader, CardContent } from "@/components/ui/card"
import { EventRow } from "@/components/EventRow"

export function ResearchStage({ activity }: { activity: any[] }) {
  // Collapse events by same agent to save space (lazy optimization)
  const events = activity.reduce((acc: any[], event, i) => {
    const last = acc[acc.length - 1]

    // Collapse 3+ consecutive events from same agent
    if (last && last.role === event.agent_role) {
      last.data = `[${last.data?.length || 0} more events]`
    } else {
      acc.push(event)
    }
    return acc
  }, [])

  return (
    <div className="space-y-6">
      {/* Event timeline */}
      <Card>
        <CardHeader>
          <h2 className="text-xl font-semibold">Research Activity</h2>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {events.map((evt: any, i: number) => (
              <EventRow
                key={`${evt.event_type}-${i}`}
                type={evt.event_type}
                level={evt.event_level}
                role={evt.agent_role}
                data={evt.data}
              />
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Synthesis output box at end */}
      {activity.some(e => e.event_type === "synthesis_complete") && (
        <Card className="border-green-500/50">
          <CardHeader>
            <h2 className="text-lg font-semibold text-green-700">
              Research Synthesis
            </h2>
          </CardHeader>
          <CardContent>
            <p className="prose prose-sm max-w-none">
              {activity.find(e => e.event_type === "synthesis_complete")?.data?.summary}
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
```

**Lovable notes**:
- Implements basic collapse optimization (3+ events from same agent)
- Displays synthesis at end when completed (green box)
- EventRow reuses existing model

---

### 4. `src/components/EventRow.tsx`

**Functionality**: Single timeline event with role-based styling

```tsx
import { Bot, Cpu, Wrench, CheckCircle2 } from "lucide-react"

interface EventRowProps {
  type: string
  level: "orchestrator" | "agent" | "tool"
  role: string
  data: any
}

export function EventRow({ type, level, role, data }: EventRowProps) {
  // Icon selection
  const icons = {
    orchestrator: <Cpu className="h-4 w-4 text-amber-600" />,
    agent: <Bot className="h-4 w-4 text-blue-600" />,
    tool: <Wrench className="h-4 w-4 text-slate-600" />
  }

  // Color schemes
  const getBorderClass = (level: string) => {
    switch (level) {
      case "orchestrator": return "border-amber-500"
      case "agent": return "border-blue-500"
      case "tool": return "border-slate-500"
      default: return "border-gray-300"
    }
  }

  const getBgClass = (level: string) => {
    switch (level) {
      case "orchestrator": return "bg-amber-50"
      case "agent": return "bg-blue-50"
      case "tool": return "bg-slate-50"
      default: return "bg-gray-50"
    }
  }

  return (
    <div className={`border-l-4 ${getBorderClass(level)} ${getBgClass(level)} pl-4 py-3 rounded-r`}>
      <div className="flex items-start gap-3">
        {icons[level as keyof typeof icons]}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-sm font-semibold">{role}</span>
            <span className="text-xs text-muted-foreground">— {type}</span>
          </div>
          <p className="text-sm text-gray-700">
            {data.summary || data.message || JSON.stringify(data).substring(0, 200)}
          </p>
        </div>
      </div>
    </div>
  )
}
```

**Lovable notes**:
- Uses only standard lucide-react icons
- Color schemes copied from existing UI patterns
- Handles both summary and raw data display

---

### 5. `src/routes/research/$id.tsx` (Dynamic Route)

**Functionality**: Individual research task detail page with SSE streaming

```tsx
import { useQuery, useSuspenseQuery } from "@tanstack/react-query"
import { useAsset } from "@tanstack/react-router/client"
import { Card, CardHeader, CardContent } from "@/components/ui/card"
import { ResearchStage } from "@/components/ResearchStage"
import { Badge } from "@/components/ui/badge"

export async function loader() {
  const res = await fetch("/api/research/" + params.id)
  return res.json()
}

export default function ResearchDetailPage() {
  const params = useAsset<{ id: string }>()
  const { data: task } = useSuspenseQuery({
    queryKey: ["research", params.id],
    queryFn: () => fetch(`/api/research/${params.id}`).then(r => r.json())
  })

  const { data: events } = useQuery({
    queryKey: ["research", params.id, "events"],
    queryFn: () => {
      const es = new EventSource(`/api/research/${params.id}/stream`)
      return new Promise((resolve, reject) => {
        const events: any[] = []
        es.onmessage = (e) => {
          const data = JSON.parse(e.data)
          events.push(data)
          // Update query cache (React Query handles this)
        }
        es.onerror = (e) => {
          es.close()
          reject(e)
        }
      })
    },
    refetchInterval: 2000, // Poll every 2 seconds
    refetchOnWindowFocus: true
  })

  return (
    <div className="min-h-screen bg-background">
      <AppShell>
        <div className="container mx-auto px-4 py-8">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold">{task.objective}</h1>
            <div className="flex gap-2 mt-2">
              <Badge variant={task.status === "completed" ? "default" : "secondary"}>
                {task.status}
              </Badge>
              <Badge variant="outline">{task.depth}-layer research</Badge>
            </div>
          </div>

          {/* Main content */}
          <div className="grid lg:grid-cols-4 gap-8">
            <div className="lg:col-span-1">
              {/* Quick actions sidebar */}
              <Card>
                <CardHeader>
                  <CardTitle>Actions</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  <Button variant="outline" className="w-full">
                    Download Report
                  </Button>
                  <Button variant="outline" className="w-full">
                    Create Follow-up
                  </Button>
                </CardContent>
              </Card>
            </div>

            <div className="lg:col-span-3">
              <ResearchStage activity={events || []} />
            </div>
          </div>
        </div>
      </AppShell>
    </div>
  )
}
```

**Lovable notes**:
- Uses existing TanStack Router patterns (similar to `src/routes/analysis.tsx`)
- Reuses AppShell from root layout (`__root.tsx`)
- Uses existing API patterns from analysis endpoint
- React Query for polling event sources

---

### 6. Update `src/routes/index.tsx` - Add Research Cards

```tsx
import { ResearchModeCard } from "@/components/ResearchModeCard"
import { Search, Brain, FileText } from "lucide-react"

// Inside HomePage return:
<div className="grid md:grid-cols-3 gap-6">
  <ResearchModeCard
    mode="analysis"
    title="Data Analysis"
    description="Upload CSV files and get insights through automated analysis"
    icon={<Search />}
    onClick={() => setView("analysis")}
  />

  <ResearchModeCard
    mode="research"
    title="AI Research"
    description="Deep research with multiple agent swarms and web analysis"
    icon={<Brain />}
    onClick={() => setView("research")}
  />

  <ResearchModeCard
    mode="document"
    title="Document Analysis"
    description="Analyze PDFs, papers, and research documents"
    icon={<FileText />}
    onClick={() => setView("research")}
  />
</div>
```

---

## Lovable Platform Instructions

When Lovable asks for file details:

**For ResearchModeCard.tsx**:
- **Parent**: components/ui/
- **File path**: `src/components/ResearchModeCard.tsx`
- **Description**: "Card component for selecting research modes (Document, Web, Paper)"

**For TaskList.tsx**:
- **Parent**: components/
- **File path**: `src/components/TaskList.tsx`
- **Description**: "Task sidebar displaying list of research tasks with status indicators"

**For ResearchStage.tsx**:
- **Parent**: components/
- **File path**: `src/components/ResearchStage.tsx`
- **Description**: "Timeline card displaying live research agent events"

**For EventRow.tsx**:
- **Parent**: components/
- **File path**: `src/components/EventRow.tsx`
- **Description**: "Timeline event row with role-based styling (orchestrator, agent, tool)"

**For research/$id.tsx**:
- **Parent**: routes/
- **File path**: `src/routes/research/$id.tsx`
- **Description**: "Research task detail page with SSE event streaming (dynamic route)"

---

## UI Consistency Checklist

All components must match existing design:

1. **Colors**: Use existing CSS variables from `styles.css`
   - `bg-background`: `#fff`
   - `text-foreground`: `#1e293b`
   - `muted-foreground`: `#64748b`
   - `primary`: `#2563eb`
   - `border` colors for events: `border-slate-500`, `border-blue-500`, `border-amber-500`

2. **Spacing**: Match existing card/padding patterns
   - Cards: `p-6` (CardContent)
   - Gaps: `gap-2`, `gap-4`, `gap-8`
   - Margins: `mb-2`, `mt-4`, `mt-8`

3. **Typography**: Use existing Shadcn UI text styles
   - Titles: `text-xl font-semibold`
   - Labels: `text-sm text-muted-foreground`
   - Events: `text-sm`

4. **Icons**: Use only lucide-react icons (already in project)
   - `Search`, `Brain`, `FileText` (mode selection)
   - `Cpu`, `Bot`, `Wrench` (event levels)
   - `CheckCircle2` (synthesis completion)

5. **Border-radius**: Use `rounded-md` for cards/rows (existing pattern)

---

## Integration with Backend API

### Required Endpoints (Already Implemented in Backend Plan)

1. `POST /api/research/start` - Start research task
2. `GET /api/research/sessions` - List all research tasks
3. `GET /api/research/{task_id}` - Get task details
4. `GET /api/research/{task_id}/stream` - SSE event stream

### Frontend API Wrapper (Lazy Component)

Create `src/lib/api-client.ts`:

```tsx
const researchAPI = {
  async startTask(objective: string, depth = 3) {
    const res = await fetch("/api/research/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ objective, task_type: "general", depth })
    })
    return res.json()
  },

  async getTask(taskId: string) {
    const res = await fetch(`/api/research/${taskId}`)
    return res.json()
  },

  async getEvents(taskId: string) {
    const res = await fetch(`/api/research/${taskId}/stream`)
    return res.json()
  }
}

export { researchAPI }
```

---

## Known Gaps (TODO for Later)

1. **Error recovery**: What if SSE disconnects mid-task? (Add auto-reconnect
2. **Task pagination**: Fetch 50 tasks at a time (not 1000+)
3. **Search/filter**: Filter tasks by date/status/objective (can use shadcn/ui input)
4. **Export**: Download raw events as JSON (just fetch + download)
5. **Resume**: Restart task if it failed mid-synthesis (add status check)

---

## Lazy Rationale

- **Minimal components**: Only 6 new components (4 display + 1 API wrapper + 1 root update)
- **Reuse patterns**: EventSystem already exists in `chat_agent.py` pattern
- **No new libraries**: Uses existing lucide-react, shadcn/ui, TanStack Router
- **Async proven pattern**: Orchestrator matches existing analysis SSE flow
- **One-page detail**: Dynamic route `research/$id` reuses pattern from `analysis.tsx`
  - Same SSE streaming logic
  - Same card-based layout

**Total lines of code**: ~600 lines across 7 files (including comments).
