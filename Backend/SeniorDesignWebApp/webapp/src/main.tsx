import * as React from "react"
import { createRoot } from "react-dom/client"
import { enableMocking } from "./mocks/enable.ts"
import "./index.css"

const Routing = React.lazy(() => import("./Routing.tsx"))

// If we're running with mock data enabled, setting that up is asynchronous,
// so we delay setting up the app until that's done.
enableMocking().then(() => {
  // createRoot and render are the core entry points for React
  createRoot(document.getElementById("root")!).render(<Routing />)
})
