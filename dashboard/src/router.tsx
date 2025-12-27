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
  path: "$epicSlug",
  component: EpicView,
})

const routeTree = rootRoute.addChildren([indexRoute, epicRoute])

export const router = createRouter({ routeTree })

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router
  }
}
