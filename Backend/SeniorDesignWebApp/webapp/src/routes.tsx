import type { RouteObject } from "react-router-dom"
import { StockApp } from "./StockApp"
import { About } from "./About"
import { Home } from "./Home"
import { API_URL } from "./constants"
import { Predictions } from "./Predictions";  

// This object describes the structure of the app: its possible screens, parameters,
// and data loading operations.
export const routes: RouteObject[] = [
  {
    path: "",
    element: <StockApp />,
    // Child routes are given a place to render within the parent via the <Outlet /> component
    children: [
      {
        path: "",
        element: <Home />,
      },
      {
        path: "about",
        element: <About />,
      },
      {
      path: "predictions",
      element: <Predictions />,
      loader: async () => {
        try {
          const response = await fetch(`${API_URL}/predictions`);
          console.log("API response status:", response.status); // log status
          const data = await response.json();
          console.log("Data received from API:", data); // log data received
          if (!response.ok) {
            throw new Error("Failed fetching predictions");
          }
          return data.predictions;  
        } catch (error) {
          throw new Error("Failed to load predictions.\n" + (error as Error).message);
        }
      },
      errorElement: <div>Failed loading predictions</div>,
    },
    ],
  },
]
