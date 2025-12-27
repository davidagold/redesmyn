import {
  createRouter,
  createRootRoute,
  createRoute,
} from "@tanstack/react-router"
import { RootLayout } from "@/components/layout/RootLayout"
import { EpicView } from "@/routes/EpicView"
import { IndexView } from "@/routes/IndexView"

const rootRoute = createRootRoute({
  component: RootLayout,
})

const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: IndexView,
})

const epicRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/$epicSlug",
  component: EpicView,
})

const edgeRoute = createRoute({
  getParentRoute: () => epicRoute,
  path: "/e/$fromNodeId/$toNodeId",
  component: () => null,
})

const nodeRoute = createRoute({
  getParentRoute: () => epicRoute,
  path: "/$nodeId",
  component: () => null,
})

const routeTree = rootRoute.addChildren([
  indexRoute,
  epicRoute.addChildren([edgeRoute, nodeRoute]),
])

export const router = createRouter({ routeTree })

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router
  }
}
