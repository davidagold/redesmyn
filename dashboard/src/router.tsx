import {
  createRouter,
  createRootRoute,
  createRoute,
  redirect,
} from "@tanstack/react-router"
import { RootLayout } from "@/components/layout/RootLayout"
import { EpicView } from "@/routes/EpicView"
import { GraphIndexView } from "@/routes/GraphIndexView"

const rootRoute = createRootRoute({
  component: RootLayout,
})

const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  beforeLoad: () => {
    throw redirect({ to: "/graph" })
  },
})

// Layout route for /graph/* - no component, just passes through to children
const graphRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "graph",
})

// Index route for /graph (no epic selected)
const graphIndexRoute = createRoute({
  getParentRoute: () => graphRoute,
  path: "/",
  component: GraphIndexView,
})

// Route for /graph/$epicSlug
const epicRoute = createRoute({
  getParentRoute: () => graphRoute,
  path: "$epicSlug",
  component: EpicView,
})

const edgeRoute = createRoute({
  getParentRoute: () => epicRoute,
  path: "e/$fromTaskId/$toTaskId",
  component: () => null,
})

const nodeRoute = createRoute({
  getParentRoute: () => epicRoute,
  path: "$taskId",
  component: () => null,
})

const routeTree = rootRoute.addChildren([
  indexRoute,
  graphRoute.addChildren([
    graphIndexRoute,
    epicRoute.addChildren([edgeRoute, nodeRoute]),
  ]),
])

export const router = createRouter({ routeTree })

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router
  }
}
