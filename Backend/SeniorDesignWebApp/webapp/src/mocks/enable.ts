export async function enableMocking() {
  if (!import.meta.env.VITE_ENABLE_MOCKS && import.meta.env.NODE_ENV !== "test") {
    return
  }

  const { worker } = await import("./browser")

  // `worker.start()` returns a Promise that resolves
  // once the Service Worker is up and ready to intercept requests.
  return worker.start()
}
