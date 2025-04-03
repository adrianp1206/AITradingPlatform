import userEvent from "@testing-library/user-event";
import { RouterProvider, createMemoryRouter } from "react-router-dom";
import { test, expect, describe } from "vitest";
import { routes } from "../routes";
import { render, screen, within } from "@testing-library/react";

describe("Symbols", () => {
    test("Displys list of symbols from symbols API", async () => {
        const router = createMemoryRouter(routes, {
            initialEntries: ["/symbols"],
        });
        render(<RouterProvider router={router} />);

        const symbolsList = await screen.findByTestId("symbols-list");
        const symbols = within(symbolsList).getAllByRole("listitem");

        expect(symbols).toHaveLength(4);
        expect(symbols[0]).toHaveTextContent("AAPL");
        expect(symbols[1]).toHaveTextContent("TSLA");
        expect(symbols[2]).toHaveTextContent("BAC");
        expect(symbols[3]).toHaveTextContent("RIVN");
    });

    test("Back returns to home", async () => {
        const router = createMemoryRouter(routes, {
            initialEntries: ["/symbols"],
        });
        render(<RouterProvider router={router} />);

        await userEvent.click(await screen.findByText("Back"));
        expect(await screen.findByTestId("home")).toBeVisible();
    });
    test("clicking symbol navigates to Price", async () => {
        const router = createMemoryRouter(routes, {
            initialEntries: ["/symbols"],
        });
        render(<RouterProvider router={router} />);

        await userEvent.click(await screen.findByText("AAPL"));

        expect(await screen.findByText("Price")).toBeVisible();
    });

    test("clicking symbol navigates to AverageVolume", async () => {
        const router = createMemoryRouter(routes, {
            initialEntries: ["/symbols"],
        });
        render(<RouterProvider router={router} />);

        await userEvent.click(await screen.findByText("AAPL"));

        expect(await screen.findByText("Average Volume")).toBeVisible();
    })
});