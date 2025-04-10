# ECE400 React Web App

This directory contains the React web app which will provide the user interface for Stock App

## Important Files

- `index.html`
  - The one and only HTML file for our entire application. In "single page apps" (SPAs), there's typically only one very small HTML file, which bootstraps the JavaScript code which builds out the rest of the structure of the app.
- `src/main.tsx`
  - The initial entry point for the app referred to by the HTML. TypeScript + React files are generally either have `.ts` or `.tsx` extensions.
- `src/main.tsx`
  - The initial entry point for the app referred to by the HTML. Bootstraps API mocking and routing.
  - TypeScript + React files are generally either have `.ts` or `.tsx` extensions. The `.tsx` extension is required if you use any JSX within the file, so it can be safest to just always use `.tsx`.
- `src/routes.tsx`
  - Defines the URL structure and determines which components get used for what routes. You'll define additional routes here.
- `src/StockApp.tsx`
  - The first React component that we provide for the app. It is mounted at the top of the routing hierarchy.
- `src/StockApp.test.tsx`
  - A unit test for the StockApp component. Unit tests may be placed in the same directory as their components.
- `src/mocks/handlers.ts`
  - You'll edit this file provide additional mock API responses.

## Important Commands

The following can all be run from `webapp/`:

- `npm install`
  - Downloads the code for all of the libraries our app depends upon. Must be run once before any of the other commands.
- `npm run dev`
  - Starts the development server
- `npm run devWithMocks`
  - Starts the development server with API mocking enabled
- `npm run test`
  - Runs the unit tests
- `npm run lint`
  - Runs the linter

## Reference Documentation

- [TypeScript](https://www.typescriptlang.org/docs/handbook/intro.html)
- [React](https://react.dev/reference/react)
- [React Testing Library](https://testing-library.com/docs/react-testing-library/intro)
- [Vitest API](https://vitest.dev/api/)
- [Mock Service Worker](https://mswjs.io/docs_)
