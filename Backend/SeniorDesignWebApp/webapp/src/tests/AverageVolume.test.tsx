import userEvent from "@testing-library/user-event"
import { RouterProvider, createMemoryRouter } from "react-router-dom"
import { test, expect, describe } from "vitest"
import { routes } from "../routes";
import { render, screen} from "@testing-library/react"

describe("Average Volume", () => {
    test("Displays the correct symbol and volume", async () => {
        const router = createMemoryRouter(routes, {
          initialEntries: ["/symbols/AAPL/averagevolume"],
        });
        render(<RouterProvider router={router} />);

        await userEvent.click(await screen.findByText("AAPL"));

        await userEvent.click(await screen.findByText("Average Volume"));
    
        expect(await screen.findByText("Average Volume for AAPL:")).toBeVisible();
        expect(await screen.findByText("100,000,001.01")).toBeVisible();
      });

      test("clicking Back returns to symbols", async () => {
        const router = createMemoryRouter(routes, {
            initialEntries: ["/symbols/AAPL/averagevolume"],
          });
          render(<RouterProvider router={router} />);
      
          await userEvent.click(await screen.findByRole("link", { name: "Back to Symbols" }));
  
          expect(await screen.findByTestId("symbols-list")).toBeVisible();
      });
})