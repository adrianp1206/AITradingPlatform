import { http, HttpResponse } from "msw"
import { setupServer } from "msw/node"
import { expect, afterEach } from "vitest"
import { cleanup } from "@testing-library/react"
import * as matchers from "@testing-library/jest-dom/matchers"
import { handlers } from "../mocks/handlers"

expect.extend(matchers)

// Set up API mocking with MSW
const server = setupServer(...handlers)

beforeAll(() => server.listen())
afterEach(() => server.resetHandlers())
afterAll(() => server.close())

afterEach(() => {
  cleanup()
})
