import { Fragment, StrictMode } from "react"
import { createRoot } from "react-dom/client"
import { RouterProvider } from "@tanstack/react-router"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { router } from "./router"
import "@xyflow/react/dist/style.css"
import "./index.css"

// StrictMode double-invokes effects in dev; allow opting out to keep animations smooth while preserving HMR.
const StrictWrapper =
  import.meta.env.DEV && import.meta.env.VITE_DISABLE_STRICT_MODE !== "1"
    ? StrictMode
    : Fragment

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      gcTime: 15 * 60 * 1000,
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
})

createRoot(document.getElementById("root")!).render(
  <StrictWrapper>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  </StrictWrapper>,
)
