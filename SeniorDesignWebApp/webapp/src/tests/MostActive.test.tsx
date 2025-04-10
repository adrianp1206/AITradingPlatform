import userEvent from "@testing-library/user-event"
import { RouterProvider, createMemoryRouter } from "react-router-dom"
import { test, expect, describe, beforeEach } from "vitest"
import { routes } from "../routes";
import { render, screen } from "@testing-library/react"

describe("MostActive", () => {
    
    beforeEach(() => {
        const router = createMemoryRouter(routes, {
            initialEntries: ["/mostactive"],
        });

        render(<RouterProvider router={router} />);
    });
    test("displays symbol from most-active API", async () => {
        const symbol = await screen.findByText("AAPL");
        expect(symbol).toBeVisible();
    });

    test("clicking back link returns you to home", async () => {
        await userEvent.click(await screen.findByText("Back"));
        expect(await screen.findByTestId("home")).toBeVisible();
    });
})