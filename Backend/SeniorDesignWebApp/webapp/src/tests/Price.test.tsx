import userEvent from "@testing-library/user-event"
import { RouterProvider, createMemoryRouter } from "react-router-dom"
import { test, expect, describe } from "vitest"
import { routes } from "../routes";
import { render, screen } from "@testing-library/react"

describe("Price", () => {
    test("displays the correct symbol and formatted price", async () => {
        const router = createMemoryRouter(routes, {
          initialEntries: ["/symbols/AAPL/price"],
        });
        render(<RouterProvider router={router} />);

        await userEvent.click(await screen.findByText("AAPL"));

        await userEvent.click(await screen.findByText("Price"));
    
        expect(await screen.findByText("Price for AAPL")).toBeVisible();
        expect(await screen.findByText("$222.72")).toBeVisible();
      });

      test("clicking Back returns to the symbols page", async () => {
        const router = createMemoryRouter(routes, {
          initialEntries: ["/symbols/AAPL/price"],
        });
        render(<RouterProvider router={router} />);
    
        await userEvent.click(await screen.findByRole("link", { name: "Back to Symbols" }));

        expect(await screen.findByTestId("symbols-list")).toBeVisible();
      });
})