import { http, HttpResponse } from "msw"
import { API_URL } from "../constants"

export const handlers = [
  http.get(`${API_URL}/symbols`, () => {
    return HttpResponse.json({
      symbols: ["AAPL", "TSLA", "BAC", "RIVN"],
    })
  }),

  http.get(`${API_URL}/most-active`, () => {
    return HttpResponse.json({
      mostActiveStock: "AAPL",
    })
  }),
  http.get(`${API_URL}/price/:symbol`, (req) => {
    const { symbol } = req.params;

    const prices = {
      AAPL: 222.72,
      TSLA: 288.53,
      BAC: 45.41,
      RIVN: 9.71,
    };

    return HttpResponse.json({
      symbol,
      price: prices[symbol as keyof typeof prices] || 0.0,
    })
  }),
  http.get(`${API_URL}/averagevolume/:symbol`, (req) => {
    const { symbol } = req.params;

    const averageVolumes = {
      AAPL: 100000001.01,
      TSLA: 100000002,
      BAC: 100000003,
      RIVN: 100000004,
    };

    return HttpResponse.json({
      symbol,
      averageVolume: averageVolumes[symbol as keyof typeof averageVolumes] || 0.0,
    });
  })
]
